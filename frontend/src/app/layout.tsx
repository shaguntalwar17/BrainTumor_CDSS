import type { Metadata } from "next";
import { Merriweather, Space_Grotesk } from "next/font/google";

import AnimatedMedicalBackground from "@/components/AnimatedMedicalBackground";
import AppSidebar from "@/components/AppSidebar";
import Footer from "@/components/Footer";

import "./globals.css";

const heading = Merriweather({ subsets: ["latin"], weight: ["700"] });
const body = Space_Grotesk({ subsets: ["latin"], weight: ["400", "500", "700"] });

export const metadata: Metadata = {
  title: "Brain MRI Tumor AI Platform",
  description:
    "Academic prototype for detection, segmentation, classification, stage estimation, explainability, longitudinal comparison, and report generation"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${body.className} ${heading.className}`}>
        <AnimatedMedicalBackground />
        <div className="min-h-screen lg:flex">
          <AppSidebar />
          <div className="flex min-h-screen flex-1 flex-col">
            <div className="flex-1">{children}</div>
            <Footer />
          </div>
        </div>
      </body>
    </html>
  );
}
