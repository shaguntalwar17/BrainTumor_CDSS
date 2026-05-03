import type { Metadata } from "next";
import { Merriweather, Space_Grotesk } from "next/font/google";

import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";

import "./globals.css";

const heading = Merriweather({ subsets: ["latin"], weight: ["700"] });
const body = Space_Grotesk({ subsets: ["latin"], weight: ["400", "500", "700"] });

export const metadata: Metadata = {
  title: "Brain MRI Tumor AI Platform",
  description: "Academic prototype for detection, segmentation, classification, explainability, longitudinal comparison, and report generation"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${body.className} ${heading.className} bg-mesh`}>
        <Navbar />
        {children}
        <Footer />
      </body>
    </html>
  );
}
