// web/src/app/gallery/page.tsx
"use client"

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import Head from 'next/head';
import { motion, AnimatePresence } from 'framer-motion';
import { usePathname } from 'next/navigation';

interface GalleryImage {
  id: string;
  image_o  : string;
  image_c: string;
  uploadDate: Date;
  modelUsed: 'ResNet18' | 'VGG16' | 'VGG16 (Quantized)';
  fileName: string;
}

interface GalleryProps {
  images?: GalleryImage[];
  onImageDelete?: (id:string) => void;
}

// Updated mock data with more stable Picsum URLs
const initialMockImages: GalleryImage[] = [
  {
    id: '1',
    image_o: '/image_ex1_o.jpeg',
    image_c: '/image_ex1_o.jpeg',
    uploadDate: new Date(2023, 0, 15),
    modelUsed: 'ResNet18',
    fileName: 'vintage-portrait.jpg'
  },
  {
    id: '2',
    image_o: '/image_ex2_o.jpeg',
    image_c: '/image_ex2_o.jpeg',
    uploadDate: new Date(2023, 1, 20),
    modelUsed: 'VGG16',
    fileName: 'mountain-scene.png'
  },
  {
    id: '3',
    image_o: '/image_ex3_o.jpeg',
    image_c: '/image_ex3_o.jpeg',
    uploadDate: new Date(2023, 2, 10),
    modelUsed: 'VGG16 (Quantized)',
    fileName: 'quick-shot.jpeg'
  },
  {
    id: '4',
    image_o: '/image_ex4_o.jpeg',
    image_c: '/image_ex4_o.jpeg',
    uploadDate: new Date(2023, 3, 5),
    modelUsed: 'ResNet18',
    fileName: 'building-facade.jpg'
  },
  {
    id: '5',
    image_o: '/image_ex5_o.jpeg',
    image_c: '/image_ex5_o.jpeg',
    uploadDate: new Date(2023, 4, 12),
    modelUsed: 'VGG16',
    fileName: 'forest-path.png'
  },
  {
    id: '6',
    image_o: '/image_ex6_o.jpeg',
    image_c: '/image_ex6_o.jpeg',
    uploadDate: new Date(2023, 5, 18),
    modelUsed: 'VGG16 (Quantized)',
    fileName: 'cityscape.jpeg'
  },
  {
    id: '7',
    image_o: '/image_ex7_o.jpeg',
    image_c: '/image_ex7_o.jpeg',
    uploadDate: new Date(2023, 6, 22),
    modelUsed: 'ResNet18',
    fileName: 'beach-sunset.jpg'
  },
  {
    id: '8',
    image_o: '/image_ex8_o.jpeg',
    image_c: '/image_ex8_o.jpeg',
    uploadDate: new Date(2023, 7, 30),
    modelUsed: 'VGG16',
    fileName: 'desert-dunes.png'
  },
  {
    id: '9',
    image_o: '/image_ex9_o.jpeg',
    image_c: '/image_ex9_o.jpeg',
    uploadDate: new Date(2023, 8, 15),
    modelUsed: 'VGG16 (Quantized)',
    fileName: 'night-sky.jpeg'
  }     
];

const formatDate = (date: Date) => {
  const d = new Date(date);
  const month = (d.getMonth() + 1).toString().padStart(2, '0');
  const day = d.getDate().toString().padStart(2, '0');
  const year = d.getFullYear();
  return `${month}/${day}/${year}`;
};

const Gallery: React.FC<GalleryProps> = ({ 
  images: imagesProp, 
  onImageDelete 
}) => {
  const pathname = usePathname();
  const navLinks = [
      { href: '/', name: 'Home' },
      { href: '/gallery', name: 'Gallery' },
      { href: '/about', name: 'About' },
  ];
  const [displayImages, setDisplayImages] = useState<GalleryImage[]>(imagesProp || initialMockImages);
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null);
  const [showOriginalInViewModal, setShowOriginalInViewModal] = useState(false);
  const [filter, setFilter] = useState<string>('all');
  
  const [viewModalOpen, setViewModalOpen] = useState(false);
  const [shareModalOpen, setShareModalOpen] = useState(false);
  
  const [copiedLink, setCopiedLink] = useState(false);

  useEffect(() => {
    setDisplayImages(imagesProp || initialMockImages);
  }, [imagesProp]);

  const filteredImages = displayImages.filter(img => {
    if (filter === 'all') return true;
    return img.modelUsed === filter;
  });

  const openViewImageModal = (image: GalleryImage) => {
    setSelectedImage(image);
    setShowOriginalInViewModal(false); // Default to colorized in view modal
    setViewModalOpen(true);
  };

  const openShareImageModal = (image: GalleryImage) => {
    setSelectedImage(image);
    setShareModalOpen(true);
  };

  const copyShareLink = () => {
    if (!selectedImage) return;
    // In a real app, you'd likely have a dedicated shareable page URL for the image
    // For now, we'll use the colorized image URL itself.
    const shareUrl = selectedImage.image_c; 
    navigator.clipboard.writeText(shareUrl).then(() => {
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 2000);
    }).catch(err => console.error('Failed to copy link:', err));
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
      if (onImageDelete) {
        onImageDelete(id);
      } else {
        setDisplayImages(currentImages => currentImages.filter(img => img.id !== id));
      }
      console.log('Delete clicked for:', id);
      if(selectedImage?.id === id) {
        setSelectedImage(null);
        setViewModalOpen(false);
        setShareModalOpen(false);
      }
    }
  };

  const getModelIcon = (model: string) => { /* ... (no change) ... */ return model === 'ResNet18' ? 'fas fa-cogs' : model === 'VGG16' ? 'fas fa-brain' : model === 'VGG16 (Quantized)' ? 'fas fa-bolt' : 'fa-palette'; };
  const getModelColor = (model: string) => { /* ... (no change) ... */ return model === 'VGG16' ? 'from-pink-500 to-rose-500' : model === 'ResNet18' ? 'from-green-400 to-teal-500' : model === 'VGG16 (Quantized)' ? 'from-purple-500 to-indigo-600' : 'from-gray-400 to-gray-600'; };
  
  const shareOnSocial = (platform: 'facebook' | 'twitter' | 'whatsapp' | 'instagram') => { /* ... (implementation from previous step, ensuring selectedImage is checked) ... */ 
    if (!selectedImage) return;
    const pageUrl = window.location.href; // Share the gallery page URL as an example
    const imageUrlForDirectShare = selectedImage.image_c; // Actual image URL
    const shareText = encodeURIComponent(`Check out this image I colorized with ChromaFlow: ${selectedImage.fileName}`);
    let url = '';

    switch (platform) {
      case 'facebook':
        url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(pageUrl)}&quote=${shareText}`; // Facebook prefers sharing a page
        break;
      case 'twitter':
        url = `https://twitter.com/intent/tweet?text=${shareText}&url=${encodeURIComponent(pageUrl)}`;
        break;
      case 'whatsapp':
        url = `https://api.whatsapp.com/send?text=${shareText}%20${encodeURIComponent(pageUrl)}`;
        break;
      case 'instagram':
        alert("To share on Instagram, please save the image and upload it through the app. The image link has been copied to your clipboard if you need it!");
        navigator.clipboard.writeText(imageUrlForDirectShare);
        // No direct web intent for posting, so we don't open a new window.
        return; 
    }
    window.open(url, '_blank');
  };

  return (
    <>
      <Head>
        <title>Your Gallery | ChromaFlow</title>
        <meta name="description" content="View, download, and share your colorized masterpieces." />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 font-[Poppins,sans-serif]">
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
        
        <main className="container mx-auto px-4 py-12">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4">
              Your <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">Gallery</span>
            </h1>
            <p className="text-gray-300 text-lg">
              View, download, and share your colorized masterpieces
            </p>
          </div>

          <div className="flex justify-center mb-10">
            {/* ... (Filter buttons JSX, no changes needed here) ... */}
            <div className="bg-gray-800 bg-opacity-70 rounded-full p-1 flex space-x-1 sm:space-x-2">
              {['all', 'ResNet18', 'VGG16', 'VGG16 (Quantized)'].map(modelFilter => (
                <button
                  key={modelFilter}
                  onClick={() => setFilter(modelFilter)}
                  className={`px-3 py-2 sm:px-5 sm:py-2.5 text-xs sm:text-sm rounded-full transition-colors duration-200 focus:outline-none
                    ${ filter === modelFilter 
                        ? 'bg-blue-500 text-white shadow-md' 
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                >
                  {modelFilter === 'all' ? 'All Models' : modelFilter}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            <AnimatePresence>
              {filteredImages.map((image) => (
                <motion.div
                  key={image.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3 }}
                  className="group relative bg-gray-800 rounded-xl overflow-hidden shadow-lg hover:shadow-blue-500/30 transition-all duration-300 flex flex-col"
                >
                  <div className="relative aspect-square overflow-hidden bg-gray-700 cursor-pointer" onClick={() => openViewImageModal(image)}>
                    <Image 
                      src={image.image_c}
                      alt={image.fileName} 
                      fill 
                      className="object-cover transition-transform duration-300 group-hover:scale-105"
                      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, (max-width: 1280px) 33vw, 25vw"
                      priority={filteredImages.indexOf(image) < 4} // Prioritize loading for first few images
                    />
                    <div className="absolute top-3 left-3"> <div className={`bg-gradient-to-r ${getModelColor(image.modelUsed)} p-2 rounded-full shadow-lg flex items-center justify-center w-8 h-8`}> <i className={`fas ${getModelIcon(image.modelUsed)} text-white text-xs`}></i> </div> </div>
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end"> <div className="p-4 text-white"> <p className="text-sm font-semibold mb-1 truncate" title={image.fileName}>{image.fileName}</p> <p className="text-xs text-gray-300"> {formatDate(image.uploadDate)} </p> </div> </div>
                  </div>

                  <div className="p-3 flex justify-between items-center bg-gray-800 border-t border-gray-700 mt-auto">
                    <div className="flex space-x-2">
                       <button
                        onClick={() => openViewImageModal(image)}
                        className="p-2 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition"
                        title="View Image"
                      >
                        <i className="fas fa-eye text-sm"></i>
                      </button>
                      <button
                        onClick={() => handleDownload(image, false)} // Download colorized
                        className="p-2 rounded-full bg-green-600 hover:bg-green-500 text-white transition"
                        title="Download Colorized"
                      >
                        <i className="fas fa-download text-sm"></i>
                      </button>
                      <button
                        onClick={() => openShareImageModal(image)}
                        className="p-2 rounded-full bg-blue-600 hover:bg-blue-500 text-white transition"
                        title="Share"
                      >
                        <i className="fas fa-share-alt text-sm"></i>
                      </button>
                    </div>
                    <button
                      onClick={() => handleDelete(image.id)}
                      className="p-2 rounded-full bg-red-600 hover:bg-red-500 text-white transition" // Changed delete icon to white on red bg
                      title="Delete"
                    >
                      <i className="fas fa-trash text-sm"></i>
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {filteredImages.length === 0 && ( /* ... (Empty state JSX, no changes) ... */ 
            <div className="text-center py-20"> <i className="fas fa-images text-6xl text-gray-600 mb-6"></i> <h3 className="text-2xl font-semibold text-gray-400 mb-3"> Your Gallery is Empty </h3> <p className="text-gray-500 max-w-md mx-auto"> {filter === 'all' ? "It looks like you haven't colorized any photos yet. Start creating to see your masterpieces here!" : `You haven't colorized any images using the ${filter} model. Try it out or select another filter.`} </p> </div>)}
        </main>

        {/* View Image Modal */}
        <AnimatePresence>
          {viewModalOpen && selectedImage && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50 backdrop-blur-sm"
              onClick={() => setViewModalOpen(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="bg-gray-800 rounded-2xl p-6 max-w-2xl w-full shadow-2xl border border-gray-700"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-white">
                    {showOriginalInViewModal ? 'Original Image Preview' : 'Colorized Image Preview'}
                  </h3>
                  <button onClick={() => setViewModalOpen(false)} className="text-gray-400 hover:text-white">
                    <i className="fas fa-times text-xl"></i>
                  </button>
                </div>
                <div className="aspect-square relative rounded-lg overflow-hidden mb-4 bg-gray-700">
                  <Image 
                    src={showOriginalInViewModal ? selectedImage.image_o : selectedImage.image_c} 
                    alt={selectedImage.fileName} 
                    fill
                    className="object-contain"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                </div>
                <div className="flex flex-col sm:flex-row justify-center items-center space-y-2 sm:space-y-0 sm:space-x-3">
                  <button
                    onClick={() => setShowOriginalInViewModal(prev => !prev)}
                    className="bg-gray-700 hover:bg-gray-600 text-white py-2.5 px-4 text-sm rounded-md transition flex items-center w-full sm:w-auto justify-center"
                  >
                    <i className="fas fa-exchange-alt mr-2"></i> Toggle {showOriginalInViewModal ? 'Colorized' : 'Original'}
                  </button>
                  <button
                    onClick={() => handleDownload(selectedImage, showOriginalInViewModal)}
                    className="bg-green-600 hover:bg-green-500 text-white py-2.5 px-4 text-sm rounded-md transition flex items-center w-full sm:w-auto justify-center"
                  >
                     <i className="fas fa-download mr-2"></i> Download Current View
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Share Modal (existing structure, ensure selectedImage is used) */}
        <AnimatePresence>
          {shareModalOpen && selectedImage && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50 backdrop-blur-sm"
              onClick={() => setShareModalOpen(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="bg-gray-800 rounded-2xl p-6 max-w-md w-full shadow-2xl border border-gray-700"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-white">Share Your Creation</h3>
                   <button onClick={() => setShareModalOpen(false)} className="text-gray-400 hover:text-white"> <i className="fas fa-times text-xl"></i> </button>
                </div>
                
                <div className="mb-4">
                  <div className="aspect-video relative rounded-lg overflow-hidden mb-2 bg-gray-700"> {/* Share modal usually shows colorized */}
                    <Image src={selectedImage.image_c} alt={selectedImage.fileName} fill className="object-cover" sizes="(max-width: 768px) 100vw, 50vw" />
                  </div>
                </div>
                <p className="text-sm text-gray-300 mb-1 text-center font-medium">{selectedImage.fileName}</p>
                <p className="text-xs text-gray-400 mb-4 text-center">Model: {selectedImage.modelUsed}</p>

                <div className="space-y-3 pt-3 border-t border-gray-700">
                  <button
                    onClick={copyShareLink}
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white py-2.5 rounded-lg flex items-center justify-center space-x-2 transition"
                  >
                    <i className={`fas ${copiedLink ? 'fa-check' : 'fa-link'}`}></i>
                    <span>{copiedLink ? 'Link Copied!' : 'Copy Shareable Link'}</span>
                  </button>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {/* Social share buttons */}
                    <button onClick={() => shareOnSocial('facebook')} className="bg-[#1877F2] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-facebook-f"></i><span>Facebook</span></button>
                    <button onClick={() => shareOnSocial('twitter')} className="bg-[#1DA1F2] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-twitter"></i><span>Twitter</span></button>
                    <button onClick={() => shareOnSocial('whatsapp')} className="bg-[#25D366] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-whatsapp"></i><span>WhatsApp</span></button>
                    <button onClick={() => shareOnSocial('instagram')} className="bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-instagram"></i><span>Instagram</span></button>
                  </div>
                </div>
                 <button
                    onClick={() => handleDelete(selectedImage.id)}
                    className="mt-5 w-full text-white hover:bg-red-500 bg-red-600 py-2.5 rounded-lg transition text-sm flex items-center justify-center space-x-1.5" // Delete button in Share Modal
                  >
                    <i className="fas fa-trash"></i><span>Delete This Image</span>
                  </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
};

export default Gallery;