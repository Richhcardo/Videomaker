from flask import Flask, request, jsonify, Response, stream_with_context
import requests

app = Flask(__name__)


@app.route("/api/download")
def download():
    """Proxy de téléchargement : force un vrai téléchargement fichier
    peu importe le navigateur, en relayant la vidéo depuis son URL
    d'origine avec un header Content-Disposition: attachment."""
    video_url = (request.args.get("url") or "").strip()
    if not video_url:
        return jsonify({"error": "Paramètre 'url' manquant."}), 400

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
            "Content-Disposition": 'attachment; filename="diaporama.mp4"'
        },
    )
