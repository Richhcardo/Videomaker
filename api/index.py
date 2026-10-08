import base64
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
from urllib.parse import quote

from flask import Flask, Blueprint, request, jsonify, Response, stream_with_context
import requests

app = Flask(__name__)


def safe_filename(name):
    """Nettoie le nom pour l'utiliser comme nom de fichier (ex: 'Scène 1' -> 'Scene-1')."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return name or "diaporama"


@app.route("/api/download")
def download():
    """Proxy de téléchargement : force un vrai téléchargement fichier
    peu importe le navigateur, en relayant la vidéo depuis son URL
    d'origine avec un header Content-Disposition: attachment."""
    video_url = (request.args.get("url") or "").strip()
    if not video_url:
        return jsonify({"error": "Paramètre 'url' manquant."}), 400

    base_name = safe_filename(request.args.get("name") or "diaporama")
    filename = f"{base_name}.mp4"

    try:
        upstream = requests.get(video_url, stream=True, timeout=30)
    except Exception as e:
        return jsonify({"error": f"Échec du téléchargement : {e}"}), 502

    if not upstream.ok:
        return jsonify({"error": f"Le fichier source a répondu {upstream.status_code}"}), 502

    content_type = upstream.headers.get("Content-Type", "video/mp4")

    def generate_chunks():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(
        stream_with_context(generate_chunks()),
        content_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{quote(filename)}"
        },
    )


# ══════════════════════════════════════════════════════════════
# NARRATION (ancien narration.py, intégré dans ce fichier)
#
# Variables d'environnement (les clés restent sur le serveur) :
#   POLLY_KEY_ID         obligatoire pour la voix (Access Key ID IAM)
#   POLLY_SECRET         obligatoire pour la voix (Secret Access Key IAM)
#   POLLY_REGION         optionnel, défaut : us-east-1
#   POLLY_VOICE          optionnel, voix par défaut : Lea (ou Remi)
#   LLM_API_KEY          clé du modèle de langage gratuit (ou GROQ_API_KEY)
#   LLM_BASE_URL         défaut : https://api.groq.com/openai/v1
#   LLM_MODEL            défaut : llama-3.3-70b-versatile
#   NARRATION_WPS        mots par seconde, défaut 2.5
#
# Attention sur Vercel : les noms AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# et AWS_REGION sont réservés, d'où les noms POLLY_* ci-dessus.
# Dépendance à ajouter dans requirements.txt : boto3
# ══════════════════════════════════════════════════════════════
bp = Blueprint('narration', __name__)

UA = 'Mozilla/5.0 (compatible; DiaporamaVideo/1.0)'

LEAD = 0.3           # silence avant la voix (s)
TAIL = 0.5           # marge de fin visée pour le texte (s)
MAX_ATEMPO = 1.15    # accélération maximale de la voix (reste naturelle)
TOLERANCE = 0.10     # écart de mots toléré autour de la cible
MAX_VIDEO_BYTES = 80 * 1024 * 1024


class NarrationTooLong(Exception):
    def __init__(self, audio_s, room_s):
        super().__init__('Narration trop longue')
        self.audio_s = audio_s
        self.room_s = room_s


# ──────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────
def _env(name, default=''):
    return os.environ.get(name, default)


def _wps():
    try:
        return float(_env('NARRATION_WPS', '2.5'))
    except ValueError:
        return 2.5


def _http(method, url, headers=None, body=None, timeout=60):
    h = {'User-Agent': UA}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _msg(raw, n=200):
    return raw[:n].decode('utf-8', 'replace') if isinstance(raw, bytes) else str(raw)[:n]


def count_words(text):
    return len(re.findall(r"[\w'’\-]+", text or '', re.UNICODE))


def target_words(seconds):
    usable = max(1.0, float(seconds) - LEAD - TAIL)
    return max(3, round(usable * _wps()))


def within(words, target):
    return abs(words - target) <= max(2, TOLERANCE * target)


def safe_url(u):
    """https uniquement, et pas d'adresse interne (évite que le serveur soit détourné)."""
    p = urllib.parse.urlparse(u or '')
    if p.scheme != 'https' or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, 443):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
    except Exception:
        return False
    return True


# ──────────────────────────────────────────────────────────────
# ffmpeg
# ──────────────────────────────────────────────────────────────
def ffmpeg_bin():
    p = shutil.which('ffmpeg')
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(ff, path):
    """Durée (s) et présence d'une piste audio, via « ffmpeg -i » (pas besoin de ffprobe)."""
    r = subprocess.run([ff, '-hide_banner', '-i', path], capture_output=True, text=True)
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', r.stderr)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    has_audio = bool(re.search(r'Stream #\d+:\d+.*Audio:', r.stderr))
    return dur, has_audio


# ──────────────────────────────────────────────────────────────
# Modèle de langage (réécriture du texte)
# ──────────────────────────────────────────────────────────────
def llm_key():
    return _env('LLM_API_KEY') or _env('GROQ_API_KEY')


def llm_rewrite(text, target):
    key = llm_key()
    if not key:
        raise RuntimeError('Clé du modèle de langage absente (LLM_API_KEY ou GROQ_API_KEY)')
    base = _env('LLM_BASE_URL', 'https://api.groq.com/openai/v1').rstrip('/')
    model = _env('LLM_MODEL', 'llama-3.3-70b-versatile')
    best, feedback = text, ''
    for _ in range(2):
        messages = [
            {'role': 'system', 'content':
                "Tu adaptes des textes de narration vidéo, dans la langue du texte d'origine. "
                "Conserve le sens, le ton, la personne grammaticale et le style. "
                "Réponds uniquement par le texte final : pas de guillemets, pas de titre, pas de commentaire."},
            {'role': 'user', 'content':
                "Réécris ce texte pour qu'il fasse exactement %d mots (à 2 mots près).%s\n\nTexte :\n%s"
                % (target, feedback, text)}
        ]
        body = json.dumps({'model': model, 'messages': messages, 'temperature': 0.4, 'max_tokens': 600}).encode()
        st, raw = _http('POST', base + '/chat/completions',
                        {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, body, 45)
        if st != 200:
            raise RuntimeError('Modèle de langage HTTP %d : %s' % (st, _msg(raw)))
        out = json.loads(raw)['choices'][0]['message']['content'].strip().strip('"«» ').strip()
        if out:
            best = out
        n = count_words(best)
        if within(n, target):
            break
        feedback = " Ta réponse précédente faisait %d mots : il en faut %d." % (n, target)
    return best


# ──────────────────────────────────────────────────────────────
# Amazon Polly (voix françaises Neural)
# ──────────────────────────────────────────────────────────────
POLLY_VOICES = {'lea': 'Lea', 'remi': 'Remi'}   # fr-FR, moteur Neural
POLLY_RATE = 16000                              # PCM : 8000 ou 16000 Hz seulement


def _polly_voice(voice_id):
    """Normalise la voix saisie (« Léa », « lea »...). Voix inconnue : voix par défaut."""
    v = unicodedata.normalize('NFKD', voice_id or '').encode('ascii', 'ignore').decode('ascii').strip().lower()
    if v in POLLY_VOICES:
        return POLLY_VOICES[v]
    return POLLY_VOICES.get(_env('POLLY_VOICE', 'Lea').strip().lower(), 'Lea')


def tts(text, voice_id):
    """Voix Amazon Polly Neural. Renvoie un fichier WAV mono 16 bits (même format qu'avant)."""
    key_id = _env('POLLY_KEY_ID')
    secret = _env('POLLY_SECRET')
    if not key_id or not secret:
        raise RuntimeError('Clés Polly absentes (POLLY_KEY_ID et POLLY_SECRET)')
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        raise RuntimeError('boto3 non installé : ajoute « boto3 » dans requirements.txt')
    client = boto3.client(
        'polly',
        region_name=_env('POLLY_REGION', 'us-east-1'),
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(connect_timeout=10, read_timeout=60, retries={'max_attempts': 2}),
    )
    try:
        r = client.synthesize_speech(
            Text=text,
            VoiceId=_polly_voice(voice_id),
            Engine='neural',
            LanguageCode='fr-FR',
            OutputFormat='pcm',
            SampleRate=str(POLLY_RATE),
        )
        pcm = r['AudioStream'].read()
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError('Polly : ' + str(e)[:200])
    if not pcm:
        raise RuntimeError('Polly : pas d\'audio retourné')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(POLLY_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_seconds(b):
    """Durée exacte (s) d'un WAV, sans ffmpeg."""
    with wave.open(io.BytesIO(b), 'rb') as w:
        return w.getnframes() / float(w.getframerate())


def make_voice(text, seconds, voice_id, auto_fit, measure=None, max_tempo=MAX_ATEMPO):
    """
    Génère la voix pour une vidéo de `seconds` secondes.
    - auto_fit : réécrit le texte si le nombre de mots s'éloigne de la cible
    - measure(mp3_bytes) -> durée réelle (s), ou None si ffmpeg est absent
    Retourne (mp3, texte_final, durée_audio, texte_réécrit).
    """
    rewritten = False
    target = target_words(seconds)
    if auto_fit and not within(count_words(text), target):
        text = llm_rewrite(text, target)
        rewritten = True
    audio = tts(text, voice_id)
    dur = measure(audio) if measure else None
    if dur:
        room = seconds - LEAD - 0.1
        if dur / room > max_tempo:
            if not auto_fit:
                raise NarrationTooLong(dur, room)
            # une seule réduction, calculée avec le débit réel de la voix
            rate = count_words(text) / dur
            text = llm_rewrite(text, max(3, int(room * rate * 0.97)))
            rewritten = True
            audio = tts(text, voice_id)
            dur = measure(audio)
            if dur / room > max_tempo:
                raise NarrationTooLong(dur, room)
    return audio, text, dur, rewritten


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@bp.route('/api/narration/status')
def status():
    return jsonify({
        'ffmpeg': bool(ffmpeg_bin()),
        'polly': bool(_env('POLLY_KEY_ID') and _env('POLLY_SECRET')),
        'llm': bool(llm_key()),
        'wps': _wps(),
        'default_voice': _env('POLLY_VOICE', 'Lea')
    })


@bp.route('/api/narration/voices')
def voices():
    return jsonify({'voices': [{'id': v, 'name': v} for v in POLLY_VOICES.values()]})


@bp.route('/api/narration/fit', methods=['POST'])
def fit():
    d = request.get_json(silent=True) or {}
    text = (d.get('text') or '').strip()
    try:
        seconds = float(d.get('seconds'))
    except (TypeError, ValueError):
        return jsonify({'error': 'seconds invalide'}), 400
    if not text:
        return jsonify({'error': 'Texte vide'}), 400
    target = target_words(seconds)
    words = count_words(text)
    if within(words, target):
        return jsonify({'text': text, 'words': words, 'target': target, 'rewritten': False})
    try:
        new = llm_rewrite(text, target)
    except Exception as e:
        return jsonify({'error': str(e)}), 502
    return jsonify({'text': new, 'words': count_words(new), 'target': target, 'rewritten': True})


def _voice_of(d):
    return (d.get('voice_id') or _env('POLLY_VOICE') or 'Lea').strip()


@bp.route('/api/narration/audio', methods=['POST'])
def audio_only():
    """Voix seule (WAV), pour un montage ailleurs. Fonctionne même sans ffmpeg."""
    d = request.get_json(silent=True) or {}
    text = (d.get('text') or '').strip()
    voice = _voice_of(d)
    if not text or not voice:
        return jsonify({'error': 'Texte et voix requis'}), 400
    try:
        seconds = float(d.get('seconds'))
    except (TypeError, ValueError):
        return jsonify({'error': 'seconds invalide'}), 400
    ff = ffmpeg_bin()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            audio, final, dur, rew = make_voice(text, seconds, voice, d.get('auto_fit', True),
                                                wav_seconds, MAX_ATEMPO if ff else 1.0)
            if ff and dur:
                room = seconds - LEAD - 0.1
                tempo = min(MAX_ATEMPO, max(1.0, dur / room))
                if tempo > 1.001:
                    src, dst = os.path.join(tmp, 'a.wav'), os.path.join(tmp, 'b.wav')
                    with open(src, 'wb') as f:
                        f.write(audio)
                    subprocess.run([ff, '-y', '-i', src, '-filter:a', 'atempo=%.4f' % tempo, dst],
                                   capture_output=True, check=True)
                    with open(dst, 'rb') as f:
                        audio = f.read()
    except NarrationTooLong as e:
        return jsonify({'error': 'too_long', 'audio_seconds': round(e.audio_s, 1), 'video_seconds': round(e.room_s, 1)}), 422
    except Exception as e:
        return jsonify({'error': str(e)}), 502
    return Response(audio, mimetype='audio/wav', headers={
        'X-Narration-Text': urllib.parse.quote(final),
        'X-Narration-Words': str(count_words(final)),
        'X-Narration-Rewritten': '1' if rew else '0'})


@bp.route('/api/narration/render', methods=['POST'])
def render():
    """Télécharge la vidéo, génère la voix, cale et monte le tout. Renvoie le MP4."""
    d = request.get_json(silent=True) or {}
    url = d.get('video_url') or ''
    text = (d.get('text') or '').strip()
    voice = _voice_of(d)
    keep_ambient = bool(d.get('keep_ambient', True))
    try:
        amb = min(1.0, max(0.0, float(d.get('ambient_volume', 0.25))))
    except (TypeError, ValueError):
        amb = 0.25
    if not text or not voice:
        return jsonify({'error': 'Texte et voix requis'}), 400
    if not safe_url(url):
        return jsonify({'error': 'URL de vidéo refusée'}), 400
    ff = ffmpeg_bin()
    if not ff:
        return jsonify({'error': 'no_ffmpeg'}), 501

    try:
        with tempfile.TemporaryDirectory() as tmp:
            vpath = os.path.join(tmp, 'in.mp4')
            npath = os.path.join(tmp, 'narration.wav')
            out = os.path.join(tmp, 'out.mp4')

            req = urllib.request.Request(url, headers={'User-Agent': UA})
            total = 0
            with urllib.request.urlopen(req, timeout=120) as r, open(vpath, 'wb') as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        return jsonify({'error': 'Vidéo trop volumineuse'}), 400
                    f.write(chunk)

            vdur, has_audio = probe(ff, vpath)
            if vdur <= 0:
                return jsonify({'error': 'Vidéo illisible'}), 502

            def measure(b):
                with open(npath, 'wb') as f:
                    f.write(b)
                return probe(ff, npath)[0]

            audio, final, adur, rew = make_voice(text, vdur, voice, d.get('auto_fit', True), measure)
            with open(npath, 'wb') as f:
                f.write(audio)

            room = vdur - LEAD - 0.1
            tempo = min(MAX_ATEMPO, max(1.0, adur / room))
            lead_ms = int(LEAD * 1000)
            voice_chain = '[1:a]atempo=%.4f,adelay=%d|%d,apad' % (tempo, lead_ms, lead_ms)
            if keep_ambient and has_audio:
                fc = ('%s[n];[0:a]volume=%.3f[a0];[a0][n]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]'
                      % (voice_chain, amb))
            else:
                fc = voice_chain + '[aout]'
            cmd = [ff, '-y', '-i', vpath, '-i', npath, '-filter_complex', fc,
                   '-map', '0:v:0', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                   '-t', '%.3f' % vdur, '-movflags', '+faststart', out]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return jsonify({'error': 'ffmpeg : ' + r.stderr[-300:]}), 502
            with open(out, 'rb') as f:
                data = f.read()
    except NarrationTooLong as e:
        return jsonify({'error': 'too_long', 'audio_seconds': round(e.audio_s, 1), 'video_seconds': round(e.room_s, 1)}), 422
    except Exception as e:
        return jsonify({'error': str(e)}), 502

    return Response(data, mimetype='video/mp4', headers={
        'X-Narration-Text': urllib.parse.quote(final),
        'X-Narration-Words': str(count_words(final)),
        'X-Narration-Rewritten': '1' if rew else '0'})


app.register_blueprint(bp)
