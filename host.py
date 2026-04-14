import http.server
import socketserver

PORT = 8080

# This is a tiny "refresh" page to show your AI feed
HTML = """
<html>
    <head><meta http-equiv="refresh" content="1"></head>
    <body style="background:#1a1a1a; color:white; text-align:center; font-family:sans-serif;">
        <h1 style="color:#4CAF50;">YOLOv8 Live AI Feed</h1>
        <div style="margin:20px;">
            <img src="stream.jpg" style="width:80%; border:3px solid #333; border-radius:10px;">
        </div>
        <p>Refreshing every second...</p>
    </body>
</html>
"""

with open("index.html", "w") as f:
    f.write(HTML)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"--- VIEW YOUR AI FEED AT http://localhost:{PORT} ---")
    httpd.serve_forever()
    