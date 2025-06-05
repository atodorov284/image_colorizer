// components/TermsOfServicePage.tsx
"use client";
import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

const TermsOfServicePage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Terms of Service | ChromaFlow</title>
        <meta name="description" content="ChromaFlow's Terms of Service for using our image colorization platform." />
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
          <h1 className="text-5xl md:text-6xl font-bold mb-12 text-center gradient-text">Terms of Service</h1>
          <div className="max-w-4xl mx-auto glass-card p-8 md:p-12 rounded-3xl">
            <p className="text-gray-300 mb-6 leading-relaxed">
              Welcome to ChromaFlow! These Terms of Service ("Terms") govern your access to and use of the ChromaFlow website, services, and applications (collectively, the "Service"). By accessing or using the Service, you agree to be bound by these Terms.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">1. Acceptance of Terms</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              By accessing or using the Service, you signify your agreement to these Terms. If you do not agree to these Terms, you may not access or use the Service.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">2. Use of the Service</h2>
            <p className="text-gray-300 mb-4 leading-relaxed">
              ChromaFlow provides an AI-powered image colorization service. You agree to use the Service only for lawful purposes and in accordance with these Terms. You are responsible for any content you upload, and you must ensure that you have the necessary rights to use and process such content.
            </p>
            <p className="text-gray-300 mb-6 leading-relaxed">
              You agree not to:
              <ul className="list-disc list-inside ml-4 mt-2 space-y-2">
                <li>Upload any content that is illegal, harmful, threatening, abusive, harassing, defamatory, vulgar, obscene, or otherwise objectionable.</li>
                <li>Upload any content that infringes upon the intellectual property rights or other rights of any third party.</li>
                <li>Use the Service for any commercial purpose without obtaining a commercial license or API access.</li>
                <li>Attempt to gain unauthorized access to any portion or feature of the Service, or any other systems or networks connected to the Service.</li>
                <li>Interfere with or disrupt the operation of the Service or the servers or networks connected to the Service.</li>
              </ul>
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">3. Intellectual Property</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              You retain all ownership rights to the images you upload to ChromaFlow. We do not claim any ownership rights over your images.
              ChromaFlow and its original content, features, and functionality are and will remain the exclusive property of ChromaFlow and its licensors. The Service is protected by copyright, trademark, and other laws of both the Netherlands and foreign countries. Our trademarks and trade dress may not be used in connection with any product or service without the prior written consent of ChromaFlow.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">4. Disclaimer of Warranties</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              The Service is provided on an "AS IS" and "AS AVAILABLE" basis. ChromaFlow makes no representations or warranties of any kind, express or implied, as to the operation of the Service or the information, content, materials, or products included on the Service. You expressly agree that your use of the Service is at your sole risk.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">5. Limitation of Liability</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              In no event shall ChromaFlow, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable for any indirect, incidental, special, consequential or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from (i) your access to or use of or inability to access or use the Service; (ii) any conduct or content of any third party on the Service; (iii) any content obtained from the Service; and (iv) unauthorized access, use or alteration of your transmissions or content, whether based on warranty, contract, tort (including negligence) or any other legal theory, whether or not we have been informed of the possibility of such damage, and even if a remedy set forth herein is found to have failed of its essential purpose.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">6. Governing Law</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              These Terms shall be governed and construed in accordance with the laws of the Netherlands, without regard to its conflict of law provisions.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">7. Changes to Terms</h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material, we will provide at least 30 days' notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.
            </p>

            <h2 className="text-3xl font-bold text-violet-400 mb-4">8. Contact Us</h2>
            <p className="text-gray-300 leading-relaxed">
              If you have any questions about these Terms, please contact us at: <a href="mailto:info@chromaflow.com" className="text-violet-400 hover:underline">info@chromaflow.com</a>
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

export default TermsOfServicePage;