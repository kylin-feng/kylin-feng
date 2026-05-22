import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: '之江智慧 - AI会议记录',
  description: '简洁高效的AI会议记录工具',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh">
      <body className={`${inter.className} claude-style`}>
        {children}
      </body>
    </html>
  )
}