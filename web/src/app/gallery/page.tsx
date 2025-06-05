"use client"

import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link'; // Added for Header navigation
import Head from 'next/head'; // Added for Font Awesome and page title
import { motion, AnimatePresence } from 'framer-motion';

interface GalleryImage {
  id: string;
  originalUrl: string;
  colorizedUrl: string;
  uploadDate: Date;
  modelUsed: 'HistoColor Pro' | 'Landscape AI' | 'TurboColor';
  fileName: string;
}

interface GalleryProps {
  images?: GalleryImage[]; // Prop for external images
  onImageDelete?: (id:string) => void; // Prop for handling delete externally
}

// Initial mock data
const initialMockImages: GalleryImage[] = [
  {
    id: '1',
    originalUrl: 'https://source.unsplash.com/random/400x400?blackandwhite,portrait',
    colorizedUrl: 'https://source.unsplash.com/random/400x400?portrait,color',
    uploadDate: new Date(2023, 0, 15),
    modelUsed: 'HistoColor Pro',
    fileName: 'vintage-portrait.jpg'
  },
  {
    id: '2',
    originalUrl: 'https://source.unsplash.com/random/400x400?blackandwhite,landscape',
    colorizedUrl: 'https://source.unsplash.com/random/400x400?landscape,color',
    uploadDate: new Date(2023, 1, 20),
    modelUsed: 'Landscape AI',
    fileName: 'mountain-scene.png'
  },
  {
    id: '3',
    originalUrl: 'https://source.unsplash.com/random/400x400?blackandwhite,abstract',
    colorizedUrl: 'https://source.unsplash.com/random/400x400?abstract,color',
    uploadDate: new Date(2023, 2, 10),
    modelUsed: 'TurboColor',
    fileName: 'quick-shot.jpeg'
  },
  {
    id: '4',
    originalUrl: 'https://source.unsplash.com/random/400x400?blackandwhite,architecture',
    colorizedUrl: 'https://source.unsplash.com/random/400x400?architecture,color',
    uploadDate: new Date(2023, 3, 5),
    modelUsed: 'HistoColor Pro',
    fileName: 'building-facade.jpg'
  }
];

// Format date consistently
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
  const [displayImages, setDisplayImages] = useState<GalleryImage[]>(imagesProp || initialMockImages);
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null);
  const [showOriginalInModal, setShowOriginalInModal] = useState(false); // For modal image toggle
  const [filter, setFilter] = useState<string>('all');
  const [shareModalOpen, setShareModalOpen] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);

  // Effect to update displayImages if imagesProp changes
  useEffect(() => {
    setDisplayImages(imagesProp || initialMockImages);
  }, [imagesProp]);


  const filteredImages = displayImages.filter(img => {
    if (filter === 'all') return true;
    return img.modelUsed === filter;
  });

  const handleDownload = async (image: GalleryImage) => {
    const urlToDownload = showOriginalInModal && selectedImage?.id === image.id ? image.originalUrl : image.colorizedUrl;
    const fileName = `${image.fileName.split('.')[0]}_${showOriginalInModal && selectedImage?.id === image.id ? 'original' : 'colorized'}.${image.fileName.split('.')[1] || 'jpg'}`;
    
    try {
      const response = await fetch(urlToDownload);
      const blob = await response.blob();
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
      console.log('Download started for:', fileName);
    } catch (error) {
      console.error('Download failed:', error);
      // Fallback for placeholder/cross-origin images:
      const link = document.createElement('a');
      link.href = urlToDownload;
      link.download = fileName;
      link.target = '_blank'; // May open in new tab if direct download fails for some URLs
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      console.log('Attempted download for:', fileName, '(may open in new tab)');
    }
  };

  const handleOpenShareModal = (image: GalleryImage) => {
    setSelectedImage(image);
    setShowOriginalInModal(false); // Default to colorized in modal
    setShareModalOpen(true);
  };

  const copyShareLink = () => {
    if (!selectedImage) return;
    const shareUrl = selectedImage.colorizedUrl; // Or a dedicated share page URL
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
        // If no external handler, modify internal state for mock purposes
        setDisplayImages(currentImages => currentImages.filter(img => img.id !== id));
      }
      console.log('Delete clicked for:', id);
      if(selectedImage?.id === id) { // Close modal if deleted image was selected
        setSelectedImage(null);
        setShareModalOpen(false);
      }
    }
  };
  
  const openImageModal = (image: GalleryImage) => {
    setSelectedImage(image);
    setShowOriginalInModal(false); // Reset to colorized when opening
    // Using the share modal structure for viewing, can be adapted later if a different view modal is needed
    setShareModalOpen(true); 
  };


  const getModelIcon = (model: string) => {
    switch (model) {
      case 'HistoColor Pro': return 'fa-crown';
      case 'Landscape AI': return 'fa-mountain';
      case 'TurboColor': return 'fa-bolt';
      default: return 'fa-palette';
    }
  };

  const getModelColor = (model: string) => {
    switch (model) {
      case 'HistoColor Pro': return 'from-pink-500 to-rose-500';
      case 'Landscape AI': return 'from-green-400 to-teal-500';
      case 'TurboColor': return 'from-purple-500 to-indigo-600';
      default: return 'from-gray-400 to-gray-600';
    }
  };
  
  const shareOnSocial = (platform: 'facebook' | 'twitter' | 'whatsapp' | 'instagram') => {
    if (!selectedImage) return;
    const imageUrl = encodeURIComponent(selectedImage.colorizedUrl); // Use actual image URL
    const shareText = encodeURIComponent(`Check out this image I colorized with ChromaFlow: ${selectedImage.fileName}`);
    let url = '';

    switch (platform) {
      case 'facebook':
        url = `https://www.facebook.com/sharer/sharer.php?u=${imageUrl}`;
        break;
      case 'twitter':
        url = `https://twitter.com/intent/tweet?text=${shareText}&url=${imageUrl}`;
        break;
      case 'whatsapp':
        url = `https://api.whatsapp.com/send?text=${shareText}%20${imageUrl}`;
        break;
      case 'instagram':
        // Instagram sharing is complex via web. This will link to Instagram.
        // A better UX might be to copy the image/link and prompt user.
        alert("To share on Instagram, please save the image and upload it through the app. Image link copied to clipboard!");
        copyShareLink(); // Copies the image URL
        url = `https://instagram.com`; // Or just do nothing after alert
        break;
    }
    window.open(url, '_blank');
  };


  return (
    <>
      <Head>
        <title>Your Gallery | ChromaFlow</title>
        <meta name="description" content="View, download, and share your colorized masterpieces." />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        {/* Assuming Poppins font is desired, like in AboutPage */}
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 py-8 font-[Poppins,sans-serif]">
        {/* Added Header */}
        <header className="py-6 mb-8 sticky top-0 z-40 bg-gray-900/70 backdrop-blur-md border-b border-gray-700">
          <div className="container mx-auto px-4 flex justify-between items-center">
            <Link href="/">
              <div className="flex items-center cursor-pointer">
                <div className="bg-gradient-to-r from-blue-500 to-purple-600 w-10 h-10 rounded-lg flex items-center justify-center mr-2">
                  <i className="fas fa-palette text-white text-xl"></i>
                </div>
                <h1 className="text-2xl font-bold text-white">Chroma<span className="text-blue-400">Flow</span></h1>
              </div>
            </Link>
            <nav className="hidden md:block">
              <ul className="flex space-x-6 items-center">
                <li><Link href="/" className="text-gray-300 hover:text-blue-400 transition">Home</Link></li>
                <li><Link href="/models" className="text-gray-300 hover:text-blue-400 transition">Models</Link></li>
                <li><Link href="/gallery" className="text-blue-400 font-semibold">Gallery</Link></li>
                <li><Link href="/about" className="text-gray-300 hover:text-blue-400 transition">About</Link></li>
              </ul>
            </nav>
            <button className="md:hidden text-gray-300 hover:text-white text-2xl">
                <i className="fas fa-bars"></i>
            </button>
          </div>
        </header>
        
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4">
              Your <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">Gallery</span>
            </h1>
            <p className="text-gray-300 text-lg">
              View, download, and share your colorized masterpieces
            </p>
          </div>

          <div className="flex justify-center mb-10">
            <div className="bg-gray-800 bg-opacity-70 rounded-full p-1 flex space-x-1 sm:space-x-2">
              {['all', 'HistoColor Pro', 'Landscape AI', 'TurboColor'].map(modelFilter => (
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
                  className="group relative bg-gray-800 rounded-xl overflow-hidden shadow-lg hover:shadow-blue-500/30 transition-all duration-300"
                >
                  <div className="relative aspect-square overflow-hidden bg-gray-700 cursor-pointer" onClick={() => openImageModal(image)}>
                    <Image 
                      src={image.colorizedUrl} // Display colorized by default
                      alt={image.fileName} 
                      fill 
                      className="object-cover transition-transform duration-300 group-hover:scale-105"
                      sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, (max-width: 1280px) 33vw, 25vw"
                      onError={(e) => e.currentTarget.src = '/api/placeholder/400/400'} // Fallback
                    />
                    
                    <div className="absolute top-3 left-3">
                      <div className={`bg-gradient-to-r ${getModelColor(image.modelUsed)} p-2 rounded-full shadow-lg flex items-center justify-center w-8 h-8`}>
                        <i className={`fas ${getModelIcon(image.modelUsed)} text-white text-xs`}></i>
                      </div>
                    </div>

                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end">
                      <div className="p-4 text-white">
                        <p className="text-sm font-semibold mb-1 truncate" title={image.fileName}>{image.fileName}</p>
                        <p className="text-xs text-gray-300">
                          {formatDate(image.uploadDate)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 flex justify-between items-center bg-gray-800 border-t border-gray-700">
                    <div className="flex space-x-2">
                       <button
                        onClick={() => openImageModal(image)}
                        className="p-2 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition"
                        title="View Image"
                      >
                        <i className="fas fa-eye text-sm"></i>
                      </button>
                      <button
                        onClick={() => handleDownload(image)} // Download colorized by default from card
                        className="p-2 rounded-full bg-green-500 bg-opacity-20 hover:bg-opacity-40 text-white transition" // Icon color changed to white
                        title="Download Colorized"
                      >
                        <i className="fas fa-download text-sm"></i>
                      </button>
                      <button
                        onClick={() => handleOpenShareModal(image)}
                        className="p-2 rounded-full bg-blue-500 bg-opacity-20 hover:bg-opacity-40 text-white transition" // Icon color changed to white
                        title="Share"
                      >
                        <i className="fas fa-share-alt text-sm"></i>
                      </button>
                    </div>
                    <button
                      onClick={() => handleDelete(image.id)}
                      className="p-2 rounded-full bg-red-500 bg-opacity-20 hover:bg-opacity-40 text-red-400 hover:text-red-300 transition"
                      title="Delete"
                    >
                      <i className="fas fa-trash text-sm"></i>
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {filteredImages.length === 0 && (
            <div className="text-center py-20">
              <i className="fas fa-images text-6xl text-gray-600 mb-6"></i>
              <h3 className="text-2xl font-semibold text-gray-400 mb-3">
                Your Gallery is Empty
              </h3>
              <p className="text-gray-500 max-w-md mx-auto">
                {filter === 'all' 
                  ? "It looks like you haven't colorized any photos yet. Start creating to see your masterpieces here!" 
                  : `You haven't colorized any images using the ${filter} model. Try it out or select another filter.`}
              </p>
              {/* Optional: Add a Link/Button to the colorizer page */}
            </div>
          )}
        </div>

        <AnimatePresence>
          {shareModalOpen && selectedImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50 backdrop-blur-sm"
              onClick={() => setShareModalOpen(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.9, opacity: 0, y: 20 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="bg-gray-800 rounded-2xl p-6 max-w-lg w-full shadow-2xl border border-gray-700"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-white">
                    {showOriginalInModal ? 'Original Image' : 'Colorized Creation'}
                  </h3>
                  <button onClick={() => setShareModalOpen(false)} className="text-gray-400 hover:text-white">
                    <i className="fas fa-times text-xl"></i>
                  </button>
                </div>
                
                <div className="mb-4 relative">
                  <div className="aspect-square relative rounded-lg overflow-hidden mb-2 bg-gray-700">
                    <Image 
                      src={showOriginalInModal ? selectedImage.originalUrl : selectedImage.colorizedUrl} 
                      alt={selectedImage.fileName} 
                      fill
                      className="object-contain" // Use contain to see whole image in modal
                      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                      onError={(e) => e.currentTarget.src = '/api/placeholder/400/400'} // Fallback
                    />
                  </div>
                  <div className="flex justify-center space-x-2 absolute bottom-4 left-1/2 -translate-x-1/2">
                     <button
                        onClick={() => setShowOriginalInModal(prev => !prev)}
                        className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-3 text-xs rounded-md transition flex items-center"
                      >
                        <i className="fas fa-exchange-alt mr-1.5"></i> Toggle {showOriginalInModal ? 'Colorized' : 'Original'}
                      </button>
                      <button
                        onClick={() => handleDownload(selectedImage)} // Download based on modal view
                        className="bg-green-500 hover:bg-green-600 text-white py-2 px-3 text-xs rounded-md transition flex items-center"
                      >
                         <i className="fas fa-download mr-1.5"></i> Download
                      </button>
                  </div>
                </div>
                
                <p className="text-sm text-gray-300 mb-1 text-center font-medium">{selectedImage.fileName}</p>
                <p className="text-xs text-gray-400 mb-4 text-center">Model: {selectedImage.modelUsed} | Uploaded: {formatDate(selectedImage.uploadDate)}</p>


                <div className="space-y-3 pt-2 border-t border-gray-700">
                   <h4 className="text-sm font-semibold text-gray-200 text-center mt-2">Share this masterpiece</h4>
                  <button
                    onClick={copyShareLink}
                    className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2.5 rounded-lg flex items-center justify-center space-x-2 transition"
                  >
                    <i className={`fas ${copiedLink ? 'fa-check' : 'fa-link'}`}></i>
                    <span>{copiedLink ? 'Link Copied!' : 'Copy Shareable Link'}</span>
                  </button>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <button onClick={() => shareOnSocial('facebook')} className="bg-[#1877F2] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-facebook-f"></i><span>Facebook</span></button>
                    <button onClick={() => shareOnSocial('twitter')} className="bg-[#1DA1F2] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-twitter"></i><span>Twitter</span></button>
                    <button onClick={() => shareOnSocial('whatsapp')} className="bg-[#25D366] hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-whatsapp"></i><span>WhatsApp</span></button>
                    <button onClick={() => shareOnSocial('instagram')} className="bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 hover:opacity-90 text-white py-2.5 rounded-lg transition flex items-center justify-center space-x-1.5 text-sm"><i className="fab fa-instagram"></i><span>Instagram</span></button>
                  </div>
                </div>
                 <button
                    onClick={() => handleDelete(selectedImage.id)}
                    className="mt-5 w-full text-red-400 hover:text-red-300 hover:bg-red-500/10 py-2.5 rounded-lg transition text-sm flex items-center justify-center space-x-1.5"
                  >
                    <i className="fas fa-trash"></i><span>Delete Image</span>
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