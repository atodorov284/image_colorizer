// web/src/app/about/page.tsx
"use client"
import React from 'react';
import Head from 'next/head';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const AboutPage: React.FC = () => {
  const pathname = usePathname();
  const navLinks = [
      { href: '/', name: 'Home' },
      { href: '/gallery', name: 'Gallery' },
      { href: '/about', name: 'About' },
  ];

  const handleScrollToTeam = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    document.getElementById('team')?.scrollIntoView({ behavior: 'smooth' });
  };

  const teamMembers = [
    {
      name: 'Sven van Loon',
      image: '/Sven.jpg',
      bio: 'Professional Docker Engineer and AI student at the University of Groningen. Co-founder of ChromaFlow.',
      socials: { linkedin: 'https://www.linkedin.com/in/sven-van-loon-750373338/', github: 'https://github.com/svens0n58' }
    },
    {
      name: 'Mika Umaña',
      image: '/Mika.jpeg',
      bio: 'AI student at the University of Groningen and co-founder of ChromaFlow.',
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
      bio: 'Artificial Intelligence Student at the University of Groingen and co-founder of ChromaFlow.',
      socials: { linkedin: 'www.linkedin.com/in/christian-kobriger-171621192', github: 'https://github.com/03chrisk'}
    }
  ];

  const techStack = [
    { name: 'React', category: 'Frontend Framework', icon: '⚛️', color: 'bg-gradient-to-br from-blue-400 to-yellow-600' },
    { name: 'PyTorch', category: 'ML Framework', icon: '🔥', color: 'bg-gradient-to-br from-white-500 to-red-500' },
    { name: 'Next.js', category: 'Full-stack Framework', icon: '▲', color: 'bg-gradient-to-br from-gray-700 to-gray-900' },
    { name: 'Tailwind', category: 'CSS Framework', icon: '🎨', color: 'bg-gradient-to-br from-green-400 to-blue-500' },
    { name: 'Docker', category: 'Containerization', icon: '🐳', color: 'bg-gradient-to-br from-blue-500 to-cyan-700' },
    { name: 'TypeScript', category: 'Programming Language', icon: '📘', color: 'bg-gradient-to-br from-orange-600 to-blue-800' }
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
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        }
        .hover-glow:hover {
          box-shadow: 0 0 30px rgba(139, 92, 246, 0.3), 0 0 15px rgba(192, 132, 252, 0.2);
        }
        .floating {
          animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
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
            <Link href="/" className="flex items-center group cursor-pointer">
                <div className="bg-gradient-to-r from-blue-500 to-pink-500 w-12 h-12 rounded-full flex items-center justify-center group-hover:shadow-lg group-hover:shadow-pink-500/25 transition-all duration-300">
                    <i className="fas fa-palette text-white text-2xl"></i>
                </div>
                <h1 className="text-3xl font-[900] ml-3">
                    Chroma<span className="text-red-400">Flow</span>
                </h1>
            </Link>
            <nav className="hidden md:block">
              <ul className="flex space-x-8">
                {navLinks.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className={`transition ${
                        pathname === link.href
                          ? 'text-red-400 font-semibold'
                          : 'text-gray-300 hover:text-red-400'
                      }`}
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
            <button className="md:hidden text-2xl text-gray-300 hover:text-white focus:outline-none">
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
                Bringing <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">History to Life</span>
              </h1>
              <a href="#team" onClick={handleScrollToTeam} className="inline-flex items-center bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white px-10 py-5 rounded-2xl text-lg font-semibold transition-all duration-300 transform hover:-translate-y-1 hover:shadow-2xl hover:shadow-purple-500/25">
                <span>Meet The Team</span>
                <i className="fas fa-arrow-down ml-2"></i>
              </a>
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
                    </div>
                    <div>
                        <h3 className="text-xl font-bold mb-6 text-gray-200">Quick Links</h3>
                        <ul className="space-y-3">
                            <li><Link href="/" className="text-gray-400 hover:text-violet-400 transition-colors">Home</Link></li>
                            <li><Link href="/gallery" className="text-gray-400 hover:text-violet-400 transition-colors">Gallery</Link></li>
                            <li><Link href="/#pricing" className="text-gray-400 hover:text-violet-400 transition-colors">Pricing</Link></li> 
                        </ul>
                    </div>
                    <div>
                        <h3 className="text-xl font-bold mb-6 text-gray-200">Company</h3>
                        <ul className="space-y-3">
                            <li><Link href="/about" className="text-gray-400 hover:text-violet-400 transition-colors">About Us</Link></li>
                            <li><a href="#" className="text-gray-400 hover:text-violet-400 transition-colors">Contact Us</a></li>
                        </ul>
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