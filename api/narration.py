"""
narration.py — Narration pour le générateur vidéo.

  1. Adapte le texte au nombre de mots / secondes de la vidéo (LLM gratuit, ex. Groq)
  2. Génère la voix (ElevenLabs)
  3. Cale la voix sur la vidéo et monte le tout avec ffmpeg

Branchement dans index.py (2 lignes) :

    from narration import bp as narration_bp
    app.register_blueprint(narration_bp)

Variables d'environnement (les clés restent sur le serveur, jamais dans le navigateur) :

    ELEVENLABS_API_KEY   obligatoire
    ELEVENLABS_VOICE_ID  optionnel (voix par défaut)
    ELEVENLABS_MODEL     optionnel (défaut : eleven_multilingual_v2)
    LLM_API_KEY          clé du modèle de langage gratuit (ou GROQ_API_KEY)
    LLM_BASE_URL         défaut : https://api.groq.com/openai/v1  (toute API compatible OpenAI)
    LLM_MODEL            défaut : llama-3.3-70b-versatile
    NARRATION_WPS        mots par seconde, défaut 2.5

ffmpeg : installé sur le serveur, sinon  pip install imageio-ffmpeg  (binaire embarqué).
"""
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from flask import Blueprint, Response, jsonify, request

bp = Blueprint('narration', __name__)

ELEVEN_BASE = 'https://api.elevenlabs.io/v1'
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
# ElevenLabs
# ──────────────────────────────────────────────────────────────
def tts(text, voice_id):
    key = _env('ELEVENLABS_API_KEY')
    if not key:
        raise RuntimeError('ELEVENLABS_API_KEY absente sur le serveur')
    url = '%s/text-to-speech/%s?output_format=mp3_44100_128' % (ELEVEN_BASE, urllib.parse.quote(voice_id, safe=''))
    body = json.dumps({'text': text, 'model_id': _env('ELEVENLABS_MODEL', 'eleven_multilingual_v2')}).encode()
    st, raw = _http('POST', url, {'xi-api-key': key, 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'}, body, 90)
    if st != 200:
        raise RuntimeError('ElevenLabs HTTP %d : %s' % (st, _msg(raw)))
    return raw


def make_voice(text, seconds, voice_id, auto_fit, measure=None):
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
        if dur / room > MAX_ATEMPO:
            if not auto_fit:
                raise NarrationTooLong(dur, room)
            # une seule réduction, calculée avec le débit réel de la voix
            rate = count_words(text) / dur
            text = llm_rewrite(text, max(3, int(room * rate * 0.97)))
            rewritten = True
            audio = tts(text, voice_id)
            dur = measure(audio)
            if dur / room > MAX_ATEMPO:
                raise NarrationTooLong(dur, room)
    return audio, text, dur, rewritten


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@bp.route('/api/narration/status')
def status():
    return jsonify({
        'ffmpeg': bool(ffmpeg_bin()),
        'elevenlabs': bool(_env('ELEVENLABS_API_KEY')),
        'llm': bool(llm_key()),
        'wps': _wps(),
        'default_voice': _env('ELEVENLABS_VOICE_ID')
    })


@bp.route('/api/narration/voices')
def voices():
    key = _env('ELEVENLABS_API_KEY')
    if not key:
        return jsonify({'voices': []})
    st, raw = _http('GET', ELEVEN_BASE + '/voices', {'xi-api-key': key}, None, 30)
    if st != 200:
        return jsonify({'voices': [], 'error': 'ElevenLabs HTTP %d' % st})
    try:
        data = json.loads(raw)
        return jsonify({'voices': [{'id': v.get('voice_id'), 'name': v.get('name')} for v in data.get('voices', [])]})
    except Exception:
        return jsonify({'voices': []})


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
    return (d.get('voice_id') or _env('ELEVENLABS_VOICE_ID')).strip()


@bp.route('/api/narration/audio', methods=['POST'])
def audio_only():
    """Voix seule (MP3), pour un montage ailleurs. Fonctionne même sans ffmpeg."""
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
            def measure(b):
                p = os.path.join(tmp, 'm.mp3')
                with open(p, 'wb') as f:
                    f.write(b)
                return probe(ff, p)[0]
            audio, final, dur, rew = make_voice(text, seconds, voice, d.get('auto_fit', True), measure if ff else None)
            if ff and dur:
                room = seconds - LEAD - 0.1
                tempo = min(MAX_ATEMPO, max(1.0, dur / room))
                if tempo > 1.001:
                    src, dst = os.path.join(tmp, 'a.mp3'), os.path.join(tmp, 'b.mp3')
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
    return Response(audio, mimetype='audio/mpeg', headers={
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
            npath = os.path.join(tmp, 'narration.mp3')
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
