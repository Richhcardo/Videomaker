import re
from urllib.parse import quote

from flask import Flask, request, jsonify, Response, stream_with_context
import requests

app = Flask(__name__)
import bp as narration_bp\napp.register_blueprint(narration_bp)\n')


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
