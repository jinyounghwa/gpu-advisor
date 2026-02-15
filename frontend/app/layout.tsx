import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "GPU Advisor — AI-Powered GPU Recommendation",
  description:
    "AI 기반 GPU 추천 시스템. 그래픽카드 모델명으로 스펙, 용도, 추천을 확인하세요.",
  keywords: ["GPU", "AI", "Advisor", "그래픽카드", "추천"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className={inter.variable}>
      <body className="noise-overlay">{children}</body>
    </html>
  );
}
