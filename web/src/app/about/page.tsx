"use client"
import React from 'react';
import Head from 'next/head';
import Image from 'next/image';
import Link from 'next/link';

const AboutPage: React.FC = () => {
  const teamMembers = [
    {
      name: 'Sven van Loon',
      image: '/Sven.jpg',
      bio: 'Future Google ML engineer, Specializing in API development and machine learning, with a focus on image processing.',
      socials: { linkedin: 'https://www.linkedin.com/in/sven-van-loon-750373338/', github: 'https://github.com/svens0n58' }
    },
    {
      name: 'Mika Umaña',
      image: '/Mika.jpeg',
      bio: 'AI Beachelor student at the University of Groningen and co-founder of ChromaFlow.',
      socials: { linkedin: 'https://www.linkedin.com/in/mika-umaña-lemus-76a485260', github: 'https://github.com/MikaMann' }
    },
    {
      name: 'Alexander Todorov',
      image: '/Alex.jpg',
      bio: 'Teaching Assistant & Student at the University of Groningen and co-founder of ChromaFlow.',
      socials: { linkedin: 'https://www.linkedin.com/in/aleksandar-todorov-26b756213/', github: 'https://github.com/atodorov284'}
    },
    {
      name: 'Christian Kobriger',
      image: '/Chirs.jpeg',
      bio: 'Award-winning UX designer specializing in AI-powered creative tools.',
      socials: { linkedin: 'www.linkedin.com/in/christian-kobriger-171621192', github: 'https://github.com/03chrisk'}
    }
  ];

  const values = [
    {
      icon: 'fas fa-heart',
      title: 'Preserve Memories',
      description: 'We believe every photograph tells a story worth preserving in its full, colorful glory.',
      color: 'from-rose-500 to-pink-500'
    },
    {
      icon: 'fas fa-shield-alt',
      title: 'Privacy First',
      description: 'Your images are your memories. We never store, share, or use them for any other purpose.',
      color: 'from-emerald-500 to-teal-500'
    },
    {
      icon: 'fas fa-rocket',
      title: 'Innovation',
      description: 'Continuously pushing the boundaries of AI to deliver better, faster, more accurate results.',
      color: 'from-violet-500 to-purple-500'
    },
    {
      icon: 'fas fa-users',
      title: 'Accessibility',
      description: 'Making advanced AI technology accessible to everyone, regardless of technical expertise.',
      color: 'from-blue-500 to-indigo-500'
    }
  ];

  const techStack = [
    { name: 'TensorFlow', category: 'ML Framework', icon: '🧠', color: 'bg-gradient-to-br from-orange-500 to-red-500' },
    { name: 'PyTorch', category: 'Deep Learning', icon: '🔥', color: 'bg-gradient-to-br from-red-500 to-pink-500' },
    { name: 'Next.js', category: 'Frontend', icon: '⚛️', color: 'bg-gradient-to-br from-gray-700 to-gray-900' },
    { name: 'Python', category: 'Backend', icon: '🐍', color: 'bg-gradient-to-br from-yellow-500 to-green-500' },
    { name: 'Kubernetes', category: 'Orchestration', icon: '☸️', color: 'bg-gradient-to-br from-blue-500 to-purple-500' },
    { name: 'Redis', category: 'Caching', icon: '💾', color: 'bg-gradient-to-br from-red-600 to-red-800' }
  ];

  const faqs = [
    {
      question: 'How does ChromaFlow work?',
      answer: 'ChromaFlow uses advanced deep learning neural networks trained on millions of color images to understand and predict realistic colors for black and white photos.'
    },
    {
      question: 'Is my data safe?',
      answer: 'Absolutely. We use end-to-end encryption and never store your images. All processing happens in real-time and images are deleted immediately after.'
    },
    {
      question: 'What makes ChromaFlow different?',
      answer: 'Our proprietary AI models are specifically trained for different image types, ensuring the best results whether you\'re colorizing portraits, landscapes, or historical documents.'
    },
    {
      question: 'Can I use ChromaFlow commercially?',
      answer: 'Yes! We offer commercial licenses and API access for businesses. Contact our sales team for custom pricing and enterprise features.'
    }
  ];

  return (
    <>
      <Head>
        <title>About ChromaFlow | AI-Powered Image Colorization</title>
        <meta name="description" content="Learn about ChromaFlow's mission to bring historical photos to life through advanced AI colorization technology." />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </Head>

      <style jsx global>{`
        body {
          font-family: 'Inter', sans-serif;
        }
        .glass-card {
          background: rgba(255, 255, 255, 0.05); /* Slightly more subtle background */
          backdrop-filter: blur(12px); /* Increased blur */
          border: 1px solid rgba(255, 255, 255, 0.1); /* Softer border */
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); /* Softer shadow for depth */
        }
        .hover-glow:hover {
          box-shadow: 0 0 30px rgba(139, 92, 246, 0.3), 0 0 15px rgba(192, 132, 252, 0.2); /* Enhanced glow */
        }
        .floating {
          animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        .gradient-text {
          background: linear-gradient(135deg, #8B5CF6 0%, #C084FC 100%); /* Adjusted gradient for better readability */
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-fill-color: transparent; /* Standard property */
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
                <li><Link href="/about" className="text-violet-400 font-semibold">About</Link></li>
              </ul>
            </nav>
            <button className="md:hidden text-2xl focus:outline-none hover:text-violet-400 transition-colors">
                <i className="fas fa-bars"></i>
            </button>
          </div>
        </header>

        {/* Hero Section */}
        <section className="relative py-20 md:py-32 overflow-hidden">
          <div className="absolute inset-0">
            <div className="absolute inset-0 bg-gradient-to-r from-violet-500/20 to-purple-500/20"></div>
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl floating"></div>
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl floating" style={{animationDelay: '3s'}}></div>
          </div>
          <div className="container mx-auto px-4 relative z-10">
            <div className="max-w-5xl mx-auto text-center">
              <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-8 leading-tight">
                Bringing <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">Color</span> to History
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
                We are pioneers in AI-driven image colorization, dedicated to reviving the past and preserving memories for generations to come.
              </p>
              <Link href="#team" className="inline-flex items-center bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white px-10 py-5 rounded-2xl text-lg font-semibold transition-all duration-300 transform hover:-translate-y-1 hover:shadow-2xl hover:shadow-purple-500/25">
                <span>Meet The Team</span>
                <i className="fas fa-arrow-down ml-2"></i>
              </Link>
            </div>
          </div>
        </section>
        
        {/* Team Section */}
        <section id="team" className="py-16 md:py-24">
          <div className="container mx-auto px-4">
            <h2 className="text-5xl font-bold mb-20 text-center">
              Meet Our <span className="gradient-text">Experts</span>
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
              {teamMembers.map((member, index) => (
                <div key={member.name} className="glass-card p-8 rounded-3xl hover-glow transition-all duration-500 transform hover:-translate-y-2 group">
                  <div className="relative w-24 h-24 mx-auto mb-6">
                    <div className="absolute inset-0 bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl blur opacity-75 group-hover:opacity-100 transition-opacity duration-300"></div>
                    <div className="relative w-full h-full rounded-2xl overflow-hidden border-2 border-white/20">
                      <Image 
                        src={member.image} 
                        alt={member.name} 
                        fill 
                        className="object-cover"
                        sizes="96px"
                      />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold mb-2 text-center">{member.name}</h3>
                  <p className="text-gray-300 text-sm mb-6 text-center leading-relaxed">{member.bio}</p>
                  <div className="flex justify-center space-x-4">
                    {member.socials.linkedin && <a href={member.socials.linkedin} className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center hover:bg-blue-600 transition-colors duration-300"><i className="fab fa-linkedin"></i></a>}
                    {member.socials.github && <a href={member.socials.github} className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center hover:bg-gray-700 transition-colors duration-300"><i className="fab fa-github"></i></a>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Our Values Section */}
        <section id="values" className="py-16 md:py-24">
            <div className="container mx-auto px-4">
              <h2 className="text-5xl font-bold mb-20 text-center">
                Our Core <span className="gradient-text">Values</span>
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
                {values.map((value, index) => (
                  <div key={value.title} className="glass-card p-8 rounded-3xl hover-glow transition-all duration-500 transform hover:-translate-y-2 text-center group">
                    <div className={`bg-gradient-to-r ${value.color} w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                      <i className={`${value.icon} text-white text-3xl`}></i>
                    </div>
                    <h3 className="text-2xl font-bold mb-4">{value.title}</h3>
                    <p className="text-gray-300 leading-relaxed">{value.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

        {/* Tech Stack Section */}
        <section className="py-20 md:py-32">
            <div className="container mx-auto px-4">
                <h2 className="text-5xl font-bold mb-20 text-center">
                  Technology We <span className="gradient-text">Use</span>
                </h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6 max-w-6xl mx-auto">
                    {techStack.map((tech, index) => (
                        <div key={tech.name} className="glass-card p-6 rounded-2xl hover-glow transition-all duration-300 transform hover:-translate-y-2 text-center group">
                            <div className={`${tech.color} w-16 h-16 rounded-xl flex items-center justify-center mx-auto mb-4 text-2xl shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                              {tech.icon}
                            </div>
                            <p className="font-semibold text-sm">{tech.name}</p>
                            <p className="text-xs text-gray-400 mt-1">{tech.category}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>

        {/* FAQ Section */}
        <section id="faq" className="py-16 md:py-24">
            <div className="container mx-auto px-4">
              <h2 className="text-5xl font-bold mb-20 text-center">
                Frequently Asked <span className="gradient-text">Questions</span>
              </h2>
              <div className="max-w-4xl mx-auto space-y-6">
                {faqs.map((faq, index) => (
                  <div key={index} className="glass-card p-8 rounded-3xl hover-glow transition-all duration-300 transform hover:-translate-y-1">
                    <h3 className="text-xl font-bold text-violet-400 mb-4">{faq.question}</h3>
                    <p className="text-gray-300 leading-relaxed text-lg">{faq.answer}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

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
                    <p>&copy; {new Date().getFullYear()} ChromaFlow. All rights reserved. Made with <i className="fas fa-heart text-violet-400"></i> in Groningen.</p>
                </div>
            </div>
        </footer>
      </div>
    </>
  );
};

export default AboutPage;