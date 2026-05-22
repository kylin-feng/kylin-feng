from flask import Flask, render_template, request, jsonify, redirect
from flask_bootstrap import Bootstrap
import os

app = Flask(__name__)
Bootstrap(app)

# 模拟视频钩子数据
video_hooks = [
    {
        'id': 1,
        'title': 'Funny Cat Compilation',
        'description': 'The best cat videos from around the world',
        'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'thumbnail': 'https://neeko-copilot.bytedance.net/api/text2image?prompt=funny%20cat%20compilation%20thumbnail&image_size=square_hd'
    },
    {
        'id': 2,
        'title': 'Amazing Travel Destinations',
        'description': 'Top 10 places to visit in 2026',
        'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'thumbnail': 'https://neeko-copilot.bytedance.net/api/text2image?prompt=amazing%20travel%20destinations%20thumbnail&image_size=square_hd'
    },
    {
        'id': 3,
        'title': 'Cooking Masterclass',
        'description': 'Learn to cook like a professional chef',
        'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'thumbnail': 'https://neeko-copilot.bytedance.net/api/text2image?prompt=cooking%20masterclass%20thumbnail&image_size=square_hd'
    }
]

@app.route('/')
def index():
    return render_template('index.html', video_hooks=video_hooks)

@app.route('/add', methods=['GET', 'POST'])
def add_hook():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        url = request.form['url']
        thumbnail = request.form['thumbnail']
        
        new_hook = {
            'id': len(video_hooks) + 1,
            'title': title,
            'description': description,
            'url': url,
            'thumbnail': thumbnail
        }
        
        video_hooks.append(new_hook)
        return redirect('/')
    return render_template('add.html')

@app.route('/api/hooks')
def get_hooks():
    return jsonify(video_hooks)

@app.route('/delete/<int:hook_id>')
def delete_hook(hook_id):
    global video_hooks
    video_hooks = [hook for hook in video_hooks if hook['id'] != hook_id]
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=5000)