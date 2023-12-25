import './ui/globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import ProvidersWrapper from './ProvidersWrapper'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ACO',
  description: 'An online environment for executing small Triton/CUDA kernels.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{<ProvidersWrapper>{children}</ProvidersWrapper>}</body>
    </html>
  )
}
