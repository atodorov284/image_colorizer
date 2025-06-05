"use client";
import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

const PrivacyPolicyPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Privacy Policy | ChromaFlow</title>
        <meta name="description" content="ChromaFlow's Privacy Policy regarding your data and image processing." />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </Head>

      <style jsx global>{`
        body {
          font-family: 'Inter', sans-serif;
        }
        .glass-card {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        }
        .gradient-text {
          background: linear-gradient(135deg, #8B5CF6 0%, #C084FC 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-fill-color: transparent;
        }
      `}</style>

      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
        {/* Header */}
        <header className="py-6 sticky top-0 z-50 bg-slate-900/80 backdrop-blur-lg border-b border-white/10">
          <div className="container mx-auto px-4 flex justify-between items-center">
            <Link href="/">
              <div className="flex items-center cursor-pointer group">
                <div className="bg-gradient-to-r from-violet-500 to-purple-600 w-12 h-12 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300">
                  <i className="fas fa-palette text-white text-xl"></i>
                </div>
                <h1 className="text-3xl font-bold ml-3">Chroma<span className="gradient-text">Flow</span></h1>
              </div>
            </Link>
            <nav className="hidden md:block">
              <ul className="flex space-x-8">
                <li><Link href="/" className="hover:text-violet-400 transition-colors duration-300">Home</Link></li>
                <li><Link href="/models" className="hover:text-violet-400 transition-colors duration-300">Models</Link></li>
                <li><Link href="/gallery" className="hover:text-violet-400 transition-colors duration-300">Gallery</Link></li>
                <li><Link href="/about" className="hover:text-violet-400 transition-colors duration-300">About</Link></li>
              </ul>
            </nav>
            <button className="md:hidden text-2xl focus:outline-none hover:text-violet-400 transition-colors">
                <i className="fas fa-bars"></i>
            </button>
          </div>
        </header>

        <main className="container mx-auto px-4 py-16 md:py-24">
          <h1 className="text-5xl md:text-6xl font-bold mb-12 text-center gradient-text">Privacy Policy</h1>
          <div className="max-w-4xl mx-auto glass-card p-8 md:p-12 rounded-3xl">
            <p className="text-gray-300 mb-6 leading-relaxed">
              At ChromaFlow, we are committed to protecting your privacy. This Privacy Policy outlines how we handle your information when you use our image colorization service.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">1. Information We Collect</h2>
            <p className="text-gray-300 mb-4 leading-relaxed">
              When you use ChromaFlow, we process the image you upload to apply colorization.
              <strong className="text-white"> We do NOT store, save, or retain any images you upload to our servers after the colorization process is complete.</strong> All image processing is done in real-time, and the image is immediately deleted from our temporary memory once the result is delivered to you.
            </p>
            <p className="text-gray-300 mb-6 leading-relaxed">
              We do not collect any personal identifying information, such as your name, email address, or IP address, unless you choose to contact us directly (e.g., via email for support or business inquiries).
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">2. How We Use Your Information</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              The only purpose for which we "use" your uploaded images is to perform the requested colorization and return the processed image to you. We do not use your images for training our AI models, for marketing purposes, or for any other commercial or non-commercial use.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">3. Data Security</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              We implement industry-standard security measures, including end-to-end encryption, to protect your images during transmission and processing. Given our policy of immediate deletion, there is no long-term storage of your images on our systems.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">4. Third-Party Services</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              ChromaFlow may utilize third-party cloud computing services for image processing. However, our strict data handling policies (immediate deletion, no storage) extend to these services as well. We choose partners who adhere to high standards of data privacy and security.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">5. Changes to This Policy</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page. You are advised to review this Privacy Policy periodically for any changes.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">6. Contact Us</h2>
            <p className="text-gray-300 leading-relaxed">
              If you have any questions about this Privacy Policy, please contact us at: <a href="mailto:info@chromaflow.com" className="text-violet-400 hover:underline">info@chromaflow.com</a>
            </p>
          </div>
        </main>

        {/* Footer */}
        <footer className="py-16 bg-slate-900/80 backdrop-blur-lg border-t border-white/10">
            <div className="container mx-auto px-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12 max-w-6xl mx-auto">
                    <div>
                        <div className="flex items-center mb-6">
                            <div className="bg-gradient-to-r from-violet-500 to-purple-600 w-12 h-12 rounded-xl flex items-center justify-center shadow-lg">
                                <i className="fas fa-palette text-white text-xl"></i>
                            </div>
                            <h1 className="text-2xl font-bold ml-3">Chroma<span className="gradient-text">Flow</span></h1>
                        </div>
                        <p className="text-gray-400 leading-relaxed">Reviving history, one color at a time, through the power of artificial intelligence.</p>
                    </div>
                    <div>
                        <h3 className="text-xl font-bold mb-6 text-gray-200">Quick Links</h3>
                        <ul className="space-y-3">
                            <li><Link href="/" className="text-gray-400 hover:text-violet-400 transition-colors">Home</Link></li>
                            <li><Link href="/models" className="text-gray-400 hover:text-violet-400 transition-colors">Our Models</Link></li>
                            <li><Link href="/gallery" className="text-gray-400 hover:text-violet-400 transition-colors">Gallery</Link></li>
                            <li><Link href="/#pricing" className="text-gray-400 hover:text-violet-400 transition-colors">Pricing</Link></li> 
                        </ul>
                    </div>
                    <div>
                        <h3 className="text-xl font-bold mb-6 text-gray-200">Company</h3>
                        <ul className="space-y-3">
                            <li><Link href="/about" className="text-gray-400 hover:text-violet-400 transition-colors">About Us</Link></li>
                            <li><Link href="/privacy-policy" className="text-gray-400 hover:text-violet-400 transition-colors">Privacy Policy</Link></li>
                            <li><Link href="/terms-of-service" className="text-gray-400 hover:text-violet-400 transition-colors">Terms of Service</Link></li>
                            <li><a href="#" className="text-gray-400 hover:text-violet-400 transition-colors">Contact Us</a></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="text-xl font-bold mb-6 text-gray-200">Connect With Us</h3>
                        <div className="flex space-x-4 mb-6">
                            <a href="#" className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center hover:bg-blue-500 transition-colors duration-300">
                                <i className="fab fa-twitter text-xl"></i>
                            </a>
                            <a href="#" className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center hover:bg-pink-500 transition-colors duration-300">
                                <i className="fab fa-instagram text-xl"></i>
                            </a>
                            <a href="#" className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center hover:bg-blue-600 transition-colors duration-300">
                                <i className="fab fa-linkedin-in text-xl"></i>
                            </a>
                            <a href="#" className="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center hover:bg-gray-700 transition-colors duration-300">
                                <i className="fab fa-github text-xl"></i>
                            </a>
                        </div>
                        <h4 className="text-lg font-bold mb-4 text-gray-300">Newsletter</h4>
                        <div className="flex">
                            <input type="email" placeholder="your.email@example.com" className="flex-1 px-4 py-3 rounded-l-xl bg-white/10 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-violet-500 backdrop-blur-sm" />
                            <button className="bg-gradient-to-r from-violet-500 to-purple-600 px-6 py-3 rounded-r-xl hover:from-violet-600 hover:to-purple-700 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-violet-500">
                                <i className="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
                <div className="border-t border-white/10 mt-12 pt-8 text-center text-gray-500">
                    <p>&copy; {new Date().getFullYear()} ChromaFlow. All rights reserved. Made with <i className="fas fa-heart text-violet-400"></i> in Silicon Valley.</p>
                </div>
            </div>
        </footer>
      </div>
    </>
  );
};

export default PrivacyPolicyPage;
