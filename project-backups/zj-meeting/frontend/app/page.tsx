'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:3001'

export default function Home() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [meetings, setMeetings] = useState([])
  const [currentMeeting, setCurrentMeeting] = useState(null)
  const [showAuth, setShowAuth] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      setIsLoggedIn(true)
      loadMeetings()
    }
  }, [])

  const loadMeetings = async () => {
    try {
      const response = await axios.get(`${API_BASE}/meetings`)
      setMeetings(response.data)
    } catch (error) {
      console.error('加载会议失败:', error)
    }
  }

  const createMeeting = async () => {
    const title = prompt('请输入会议标题:')
    if (!title) return

    try {
      const response = await axios.post(`${API_BASE}/meetings/create?title=${title}`)
      setCurrentMeeting(response.data)
      loadMeetings()
    } catch (error) {
      alert('创建会议失败')
    }
  }

  const uploadAudio = async (meetingId: number) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'audio/*'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      const formData = new FormData()
      formData.append('file', file)

      try {
        await axios.post(`${API_BASE}/meetings/${meetingId}/upload-audio`, formData)
        alert('音频上传成功，正在转录...')
        setTimeout(() => loadMeetingDetail(meetingId), 2000)
      } catch (error) {
        alert('上传失败')
      }
    }
    input.click()
  }

  const analyzeMeeting = async (meetingId: number) => {
    try {
      await axios.post(`${API_BASE}/meetings/${meetingId}/analyze`)
      alert('分析完成')
      loadMeetingDetail(meetingId)
    } catch (error) {
      alert('分析失败')
    }
  }

  const loadMeetingDetail = async (meetingId: number) => {
    try {
      const response = await axios.get(`${API_BASE}/meetings/${meetingId}`)
      setCurrentMeeting(response.data)
    } catch (error) {
      console.error('加载会议详情失败:', error)
    }
  }

  if (!isLoggedIn) {
    return <AuthComponent onSuccess={() => { setIsLoggedIn(true); loadMeetings() }} />
  }

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">之江智慧</h1>
              <span className="ml-3 text-sm text-gray-500">AI会议记录</span>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={createMeeting}
                className="bg-primary text-white px-4 py-2 rounded-md hover:bg-blue-700 transition"
              >
                新建会议
              </button>
              <button
                onClick={() => { localStorage.removeItem('token'); setIsLoggedIn(false) }}
                className="text-gray-500 hover:text-gray-700"
              >
                退出
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 会议列表 */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">会议列表</h2>
              <div className="space-y-3">
                {meetings.map((meeting: any) => (
                  <div
                    key={meeting.id}
                    onClick={() => loadMeetingDetail(meeting.id)}
                    className="p-3 border border-gray-100 rounded-lg hover:bg-gray-50 cursor-pointer transition"
                  >
                    <div className="font-medium text-gray-900">{meeting.title}</div>
                    <div className="text-sm text-gray-500">
                      {new Date(meeting.created_at).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 会议详情 */}
          <div className="lg:col-span-2">
            {currentMeeting ? (
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-900">
                    {currentMeeting.title}
                  </h2>
                  <div className="space-x-3">
                    <button
                      onClick={() => uploadAudio(currentMeeting.id)}
                      className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition"
                    >
                      上传音频
                    </button>
                    {currentMeeting.transcript && (
                      <button
                        onClick={() => analyzeMeeting(currentMeeting.id)}
                        className="bg-primary text-white px-4 py-2 rounded-md hover:bg-blue-700 transition"
                      >
                        AI分析
                      </button>
                    )}
                  </div>
                </div>

                {/* 转录内容 */}
                {currentMeeting.transcript && (
                  <div className="mb-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-3">会议转录</h3>
                    <div className="bg-gray-50 rounded-lg p-4 text-gray-700">
                      {currentMeeting.transcript}
                    </div>
                  </div>
                )}

                {/* AI分析结果 */}
                {currentMeeting.analysis && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">会议总结</h3>
                      <div className="bg-blue-50 rounded-lg p-4 text-gray-700">
                        {currentMeeting.analysis.summary}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">关键要点</h3>
                      <ul className="bg-gray-50 rounded-lg p-4 space-y-2">
                        {currentMeeting.analysis.key_points?.map((point: string, index: number) => (
                          <li key={index} className="text-gray-700">• {point}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">任务清单</h3>
                      <div className="space-y-3">
                        {currentMeeting.analysis.tasks?.map((task: any, index: number) => (
                          <div key={index} className="bg-yellow-50 rounded-lg p-4 border-l-4 border-yellow-400">
                            <div className="font-medium text-gray-900">{task.task}</div>
                            <div className="text-sm text-gray-600 mt-1">
                              负责人: {task.assignee} | 截止: {task.deadline}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">行动项</h3>
                      <ul className="bg-green-50 rounded-lg p-4 space-y-2">
                        {currentMeeting.analysis.action_items?.map((item: string, index: number) => (
                          <li key={index} className="text-gray-700">✓ {item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-lg border border-gray-200 p-6 text-center">
                <div className="text-gray-500 text-lg">选择一个会议查看详情，或创建新会议</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function AuthComponent({ onSuccess }: { onSuccess: () => void }) {
  const [isLogin, setIsLogin] = useState(true)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const handleAuth = async () => {
    try {
      if (isLogin) {
        const response = await axios.post(`${API_BASE}/auth/login?username=${username}&password=${password}`)
        localStorage.setItem('token', response.data.access_token)
        onSuccess()
      } else {
        await axios.post(`${API_BASE}/auth/register?username=${username}&email=${email}&password=${password}`)
        alert('注册成功，请登录')
        setIsLogin(true)
      }
    } catch (error: any) {
      alert(error.response?.data?.detail || '操作失败')
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">之江智慧</h1>
          <p className="mt-2 text-gray-600">AI会议记录工具</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-8">
          <div className="space-y-4">
            <input
              type="text"
              placeholder="用户名"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
            />
            {!isLogin && (
              <input
                type="email"
                placeholder="邮箱"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              />
            )}
            <input
              type="password"
              placeholder="密码"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <button
              onClick={handleAuth}
              className="w-full bg-primary text-white py-3 rounded-md hover:bg-blue-700 transition"
            >
              {isLogin ? '登录' : '注册'}
            </button>
            <div className="text-center">
              <button
                onClick={() => setIsLogin(!isLogin)}
                className="text-primary hover:underline"
              >
                {isLogin ? '没有账号？注册' : '已有账号？登录'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}