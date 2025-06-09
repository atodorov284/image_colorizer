'use client';

import React, { useState, useEffect, useRef } from 'react';


// Global type definitions
declare global {
  interface Window {
    tailwind: {
      config: any;
    };
  }
}

// Component interfaces
interface ModelCard {
  id: string;
  name: string;
  description: string;
  icon: string;
  gradient: string;
  isRecommended?: boolean;
  isBase?: boolean; // Optional property to indicate if it's a base model
}

interface ProcessingState {
  isProcessing: boolean;
  showResults: boolean;
}

const ChromaFlow: React.FC = () => {
  // State management
  const [selectedModel, setSelectedModel] = useState<string>('histocolor');
  const [processingState, setProcessingState] = useState<ProcessingState>({
    isProcessing: false,
    showResults: false
  });
  const [sliderValue, setSliderValue] = useState<number>(50);
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [originalImageSrc, setOriginalImageSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const colorizedImageRef = useRef<HTMLImageElement>(null);
  const sliderHandleRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Model data
  const models: ModelCard[] = [
    {
      id: 'ResNet18',
      name: 'ResNet18 (Fine-tuned)',
      description: 'Fine-tuned ResNet18 model, using a simplistic approach (MSE)',
      icon: 'fas fa-cogs',
      gradient: 'from-green-400 to-teal-500',
      isBase: true
    },
    {
      id: 'VGG16',
      name: 'VGG16',
      description: 'Fine-tuned VGG16, using quantile weight-rebalancing L1 loss, learning-rate scheduler, and AdamW',
      icon: 'fas fa-brain',
      gradient: 'from-red-400 to-pink-500',
      isRecommended: true
    },
    {
      id: 'VGG16 (Quantized)',
      name: 'VGG16 (Quantized)',
      description: 'Dynamically quantized VGG16-INT8 model for lower resource usage',
      icon: 'fas fa-bolt',
      gradient: 'from-purple-500 to-indigo-600'
    }
  ];

  // Update slider position
  const updateSliderPosition = () => {
    if (colorizedImageRef.current && sliderHandleRef.current) {
      sliderHandleRef.current.style.left = `${sliderValue}%`;
      colorizedImageRef.current.style.clipPath = `polygon(0 0, ${sliderValue}% 0, ${sliderValue}% 100%, 0% 100%)`;
    }
  };

  useEffect(() => {
    updateSliderPosition();
  }, [sliderValue]);

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleFileSelected = async (file: File | undefined) => {
  if (file) {
    if (file.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB. Please upload a smaller image.");
      setUploadedFile(null);
      setOriginalImageSrc(null);
      return;
    }
    if (!['image/jpeg', 'image/png', 'image/bmp'].includes(file.type)) {
      setError("Invalid file type. Please upload JPG, PNG, or BMP.");
      setUploadedFile(null);
      setOriginalImageSrc(null);
      return;
    }

    try {
      // Get the grayscale preview from the API
      const formData = new FormData();
      formData.append('image', file);
      
      const response = await fetch('http://localhost:8000/preview', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to get image preview');
      }

      const previewBlob = await response.blob();
      const previewUrl = URL.createObjectURL(previewBlob);
      
      setUploadedFile(file);
      setOriginalImageSrc(previewUrl);
      setError(null);
      setProcessingState({ isProcessing: false, showResults: false });
      if (colorizedImageRef.current) {
        colorizedImageRef.current.src = "";
      }
    } catch (error) {
      setError("Error processing image. Please try again.");
      console.error("Error processing image:", error);
    }
  }
};

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    const file = e.dataTransfer.files?.[0];
    handleFileSelected(file);
    // console.log('Files dropped:', e.dataTransfer.files);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    handleFileSelected(file);
  };

  // Colorize button handler
  const handleColorize = async () => {
    if (!uploadedFile) {
      setError("Please upload an image first.");
      return;
    }
    setError(null);
    setProcessingState({ isProcessing: true, showResults: false });

    const getApiModelType = (modelId: string): string => {
      if (modelId === 'VGG16') return 'VGG16';      // VGG16 (Quantized) (Fine-tuned)
      if (modelId === 'ResNet18') return 'resnet';   // ResNet18 (Fine-tuned)
      if (modelId === 'VGG16 (Quantized)') return 'VGG16 (Quantized)';     // VGG16 (Quantized) (Quantized)
      return 'VGG16'; // Default to VGG16
    };

    const formData = new FormData();
    formData.append('image', uploadedFile);

    try {
      const apiModelType = getApiModelType(selectedModel);
      const response = await fetch(`http://localhost:8000/predict?model=${apiModelType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error occurred during colorization." }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const imageBlob = await response.blob();
      if (colorizedImageRef.current) {
        colorizedImageRef.current.src = URL.createObjectURL(imageBlob);
      }
      setProcessingState({ isProcessing: false, showResults: true });
      // Scroll to results
      document.getElementById('resultContainer')?.scrollIntoView({ behavior: 'smooth' });
    } catch (err: any) {
      console.error("Error colorizing image:", err);
      setError(err.message || "Failed to colorize image. Please try again.");
      setProcessingState({ isProcessing: false, showResults: false });
    }
  };

  return (
    <>
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen text-white font-sans">
        {/* Header */}
        <header className="py-6">
          <div className="container mx-auto px-4 flex justify-between items-center">
            <div className="flex items-center">
              <div className="bg-gradient-to-r from-blue-500 to-pink-500 w-12 h-12 rounded-full flex items-center justify-center">
                <i className="fas fa-palette text-white text-2xl"></i>
              </div>
              <h1 className="text-3xl font-[900] ml-3">
                Chroma<span className="text-red-400">Flow</span>
              </h1>
            </div>
            <nav className="hidden md:block">
              <ul className="flex space-x-8">
                <li><a href="#" className="hover:text-red-400 transition">Home</a></li>
                <li><a href="/models" className="hover:text-red-400 transition">Models</a></li>
                <li><a href="/gallery" className="hover:text-red-400 transition">Gallery</a></li>
                <li><a href="/about" className="hover:text-red-400 transition">About</a></li>
              </ul>
            </nav>
            <button className="md:hidden text-2xl">
              <i className="fas fa-bars"></i>
            </button>
          </div>
        </header>

        {/* Hero Section */}
        <section className="py-16 text-center bg-gradient-to-r from-blue-500/10 to-pink-500/10">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-5xl md:text-6xl font-[1000] mb-6">
                Exploring Deep Learning Image <span className="text-red-400">Colorization</span>
              </h2>
              <p className="text-xl text-gray-300 mb-10">
                Image colorization using different deep learning models.
              </p>
              <div className="flex flex-col md:flex-row justify-center space-y-4 md:space-y-0 md:space-x-6">
                <div className="bg-gradient-to-r from-blue-500 to-pink-500 px-8 py-4 rounded-full text-lg font-semibold transition transform">
                  Try Colorizing an Image Below
                </div>
                <a href="/gallery">
                  <button className="bg-slate-600 hover:bg-opacity-80 px-8 py-4 rounded-full text-lg font-semibold transition transform hover:-translate-y-1">
                    <i className="fas fa-images mr-2"></i> View Examples
                  </button>
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* Main Colorizer Tool */}
        <section className="py-16">
          <div className="container mx-auto px-4">
            <div className="max-w-4xl mx-auto bg-slate-900 rounded-2xl p-8 bg-gradient-to-r from-blue-500/5 to-pink-500/5">
              <h2 className="text-3xl font-bold mb-8 text-center">Image Colorizer Tool</h2>

              {/* Model Selection */}
              <div className="mb-10">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <i className="fas fa-robot mr-2 text-red-400"></i> Select AI Model
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {models.map((model) => (
                    <div
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`cursor-pointer bg-slate-800/50 p-6 rounded-xl border-2 transition-all transform hover:-translate-y-1 hover:shadow-lg ${
                        selectedModel === model.id 
                          ? 'border-red-400 shadow-lg shadow-red-400/20' 
                          : 'border-transparent hover:border-red-400'
                      }`}
                    >
                      <div className="text-center">
                        <div className={`bg-gradient-to-r ${model.gradient} w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4`}>
                          <i className={`${model.icon} text-white text-2xl`}></i>
                        </div>
                        <h4 className="font-semibold text-lg mb-2">{model.name}</h4>
                        <p className="text-gray-300 text-sm">{model.description}</p>
                        {model.isRecommended && (
                          <div className="mt-4">
                            <span className="text-sm bg-red-400/30 py-1 px-3 rounded-full">Recommended Model</span>
                          </div>
                        )}
                        {model.isBase && (
                          <div className="mt-4">
                            <span className="text-sm bg-red-100/30 py-1 px-3 rounded-full">Base Model</span>
                          </div>
                        
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Upload Section */}
              <div className="mb-10">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <i className="fas fa-cloud-upload-alt mr-2 text-red-400"></i> Upload Your Image
                </h3>
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`bg-slate-700/20 rounded-2xl p-12 text-center border-2 border-dashed transition-all ${
                    isDragActive 
                      ? 'border-red-400 bg-red-400/10' 
                      : 'border-slate-600'
                  }`}
                >
                  <i className="fas fa-cloud-upload-alt text-slate-400 text-5xl mb-4"></i>
                  <p className="text-xl font-medium mb-2">Drag & drop your photo here</p>
                  <p className="text-gray-400 mb-6">or</p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                    accept="image/jpeg,image/png,image/bmp"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-gradient-to-r from-blue-500 to-pink-500 hover:opacity-90 px-6 py-3 rounded-full text-lg font-semibold transition"
                  >
                    <i className="fas fa-search mr-2"></i> Browse Files
                  </button>
                  <p className="text-gray-500 mt-4 text-sm">Supported Formats: JPG, PNG, BMP (max 10MB)</p>
                </div>

                {uploadedFile && (
                  <div className="mt-6 text-center p-4 bg-slate-700/30 rounded-lg">
                    <p className="text-green-400 font-medium">
                      <i className="fas fa-check-circle mr-2"></i>
                      Image selected: {uploadedFile.name}
                    </p>
                    <p className="text-gray-400 text-sm">
                      ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  </div>
                )}

                {error && (
                  <div className="mt-4 text-center p-3 bg-red-500/20 rounded-lg">
                    <p className="text-red-400"><i className="fas fa-exclamation-triangle mr-2"></i>{error}</p>
                  </div>
                )}

                <div className="flex flex-wrap gap-3 mt-4 justify-center">
                  {[
                    { icon: 'fas fa-user', label: 'Portraits' },
                    { icon: 'fas fa-building', label: 'Architecture' },
                    { icon: 'fas fa-tree', label: 'Landscapes' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center text-sm">
                      <div className="bg-gray-600 w-8 h-8 rounded mr-2 flex items-center justify-center">
                        <i className={`${item.icon} text-xs`}></i>
                      </div>
                      {item.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* Colorize Button */}
              <div className="text-center my-12">
                <button
                  onClick={handleColorize}
                  disabled={processingState.isProcessing || !uploadedFile}
                  className="bg-gradient-to-r from-blue-500 to-pink-500 hover:opacity-90 px-12 py-4 rounded-full text-xl font-semibold transition transform hover:-translate-y-1 relative shadow-lg shadow-red-400/30 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <i className="fas fa-magic mr-2"></i> Colorize Image
                  {processingState.isProcessing && (
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                    </div>
                  )}
                </button>
              </div>

              {/* Result Comparison */}
              <div className="mt-16">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <i className="fas fa-sliders-h mr-2 text-red-400"></i> Compare Results
                </h3>
                <p className="text-gray-400 mb-6">Drag the slider to compare original and colorized version</p>

                {processingState.showResults || originalImageSrc ? (
                  <div id="resultContainer">
                    <div className="relative w-full overflow-hidden rounded-xl">
                      <img
                      src={originalImageSrc || "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=800&q=80"} // Fallback if originalImageSrc is null
                        alt="Original"
                        className="w-full h-auto block rounded-xl"
                      />
                      <img
                        ref={colorizedImageRef}
                        src={processingState.showResults && colorizedImageRef.current?.src ? colorizedImageRef.current.src : undefined}
                        alt="Colorized"
                        className="absolute top-0 left-0 w-full h-auto block rounded-xl"
                        style={{ clipPath: `polygon(0 0, ${sliderValue}% 0, ${sliderValue}% 100%, 0% 100%)` }}
                      />

                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={sliderValue}
                        onChange={(e) => setSliderValue(Number(e.target.value))}
                        className="absolute top-0 left-0 w-full h-full opacity-0 cursor-ew-resize z-10"
                      />

                      <div
                        ref={sliderHandleRef}
                        className="absolute top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-lg z-20 cursor-pointer"
                        style={{ left: `${sliderValue}%`, transform: 'translate(-50%, -50%)' }}
                      >
                        <i className="fas fa-grip-lines text-gray-700 transform rotate-90"></i>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-4 mt-6 justify-center">
                      <button className="bg-gradient-to-r from-green-500 to-teal-500 hover:opacity-90 px-6 py-3 rounded-full font-medium">
                        <i className="fas fa-download mr-2"></i> Download Colorized
                      </button>
                      <button className="bg-slate-600 hover:bg-opacity-80 px-6 py-3 rounded-full font-medium">
                        <i className="fas fa-redo mr-2"></i> Try Another Model
                      </button>
                      <button className="bg-red-400 hover:opacity-90 px-6 py-3 rounded-full font-medium">
                        <i className="fas fa-share-alt mr-2"></i> Share Result
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-700/20 rounded-2xl p-16 text-center border border-dashed border-slate-600">
                    <i className="fas fa-images text-slate-400 text-5xl mb-6"></i>
                    <h4 className="text-xl font-semibold mb-3">Your Colorized Photo Appears Here</h4>
                    <p className="text-gray-500">Upload a photo and click "Colorize Image" to see the magic</p>
                    {error && (
                       <p className="text-red-400 mt-3"><i className="fas fa-exclamation-triangle mr-2"></i>{error}</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-16 bg-gradient-to-b from-slate-800 to-slate-900">
          <div className="container mx-auto px-4">
            <h2 className="text-4xl font-[1000] mb-16 text-center">Project Highlights</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 max-w-6xl mx-auto">
              {[
                {
                  icon: 'fas fa-layer-group',
                  title: 'Multiple Model Architectures',
                  description: 'Explore colorization with different model backbones like VGG16 and ResNet.',
                  gradient: 'from-blue-500 to-pink-500'
                },
                {
                  icon: 'fas fa-bolt',
                  title: 'Efficient Processing',
                  description: 'Includes a quantized VGG16 model for faster inference, demonstrating a trade-off between speed and precision.',
                  gradient: 'from-red-400 to-orange-500'
                },
                {
                  icon: 'fas fa-palette',
                  title: 'Interactive Comparison',
                  description: 'Visually compare the original black and white image with the colorized output using an interactive slider.',
                  gradient: 'from-purple-500 to-indigo-600'
                }
              ].map((feature, index) => (
                <div key={index} className="bg-slate-900/70 p-8 rounded-2xl border border-slate-600/30 transition-all transform hover:-translate-y-1 hover:shadow-lg">
                  <div className={`bg-gradient-to-r ${feature.gradient} w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6`}>
                    <i className={`${feature.icon} text-white text-3xl`}></i>
                  </div>
                  <h3 className="text-2xl font-bold mb-3 text-center">{feature.title}</h3>
                  <p className="text-gray-400 text-center">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 bg-slate-900">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 max-w-6xl mx-auto">
              <div>
                <div className="flex items-center mb-4">
                  <div className="bg-gradient-to-r from-blue-500 to-pink-500 w-10 h-10 rounded-full flex items-center justify-center">
                    <i className="fas fa-palette text-white text-lg"></i>
                  </div>
                  <h1 className="text-2xl font-bold ml-2">Chroma<span className="text-red-400">Flow</span></h1>
                </div>
                <p className="text-gray-500">Bringing memories to life through the power of artificial intelligence and color science.</p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
                <ul>
                  {[
                    { name: 'Home', href: '#' },
                    { name: 'Gallery', href: '/gallery' },
                    { name: 'About', href: '/about' },
                  ].map((link) => (
                    <li key={link.name}>
                      <a href={link.href} className="text-gray-400 hover:text-red-400 transition">
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-4">Resources</h3>
                <ul className="space-y-2">
                  {[{name: 'GitHub Repository', href:'https://github.com/atodorov284/image_colorizer.git'}, 
                   {name:'Model Details', href:'/models'},
                   {name: 'Terms of Service', href: '/terms-of-service'},
                   {name: 'Privacy Policy', href: '/privacy-policy'}
                  ].map((link) => (
                    <li key={link.name}>
                      <a href={link.href} className="text-gray-400 hover:text-red-400 transition">{link.name}</a>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="border-t border-gray-800 mt-10 pt-6 text-center text-gray-600">
              <p>
                &copy; {new Date().getFullYear()} ChromaFlow. Made with{" "}
                <i className="fas fa-heart text-violet-400"></i> in Groningen.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default ChromaFlow;