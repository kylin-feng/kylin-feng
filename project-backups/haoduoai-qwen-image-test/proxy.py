#!/usr/bin/env python3
"""
简单代理服务器 - 解决 DashScope CORS 限制
运行方式: python3 proxy.py
代理监听: http://localhost:3000
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

TARGET = "https://dashscope.aliyuncs.com"

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[代理] {args[0]} {args[1]}")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        forward_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "origin", "referer", "accept-encoding")
        }

        req = urllib.request.Request(
            TARGET + self.path,
            data=body,
            headers=forward_headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

if __name__ == "__main__":
    server = HTTPServer(("localhost", 3000), ProxyHandler)
    print("✅ 代理已启动: http://localhost:3000")
    print("   关闭请按 Ctrl+C")
    server.serve_forever()
