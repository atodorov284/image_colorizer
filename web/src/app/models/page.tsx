"use client";
import React from "react";
import Link from "next/link";

const models = [
  {
    name: "Baseline CNN",
    description:
      "A simple convolutional neural network trained on grayscale to color mappings. Used as a performance benchmark.",
    hyperparams: {
      optimizer: "Adam",
      learningRate: "0.001",
      batchSize: "64",
      epochs: "20",
    },
  },
  {
    name: "ResNet-34 Colorizer",
    description:
      "A ResNet34 architecture modified for colorization, pretrained on ImageNet and fine-tuned on our dataset.",
    hyperparams: {
      optimizer: "AdamW",
      learningRate: "1e-4",
      batchSize: "32",
      epochs: "25",
      weightDecay: "1e-5",
    },
  },
  {
    name: "U-Net with Perceptual Loss",
    description:
      "A U-Net with VGG-based perceptual loss to ensure more vivid and semantically accurate colors.",
    hyperparams: {
      optimizer: "AdamW",
      learningRate: "1.5e-4",
      batchSize: "16",
      dropout: "0.3",
    },
  },
];
const ModelPage = () => {
  return (
    <>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
        {/* Header */}
        <header className="py-6 sticky top-0 z-50 bg-slate-900/80 backdrop-blur-lg border-b border-white/10">
          <div className="container mx-auto px-4 flex justify-between items-center">
            <Link href="/">
              <div className="flex items-center cursor-pointer group">
                <div className="bg-gradient-to-r from-violet-500 to-purple-600 w-12 h-12 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300">
                  <i className="fas fa-palette text-white text-xl"></i>
                </div>
                <h1 className="text-3xl font-bold ml-3">
                  Chroma<span className="gradient-text">Flow</span>
                </h1>
              </div>
            </Link>
            <nav className="hidden md:block">
              <ul className="flex space-x-8">
                <li>
                  <Link href="/" className="hover:text-violet-400 transition-colors duration-300">
                    Home
                  </Link>
                </li>
                <li>
                  <Link href="/models" className="text-violet-400 font-semibold">
                    Models
                  </Link>
                </li>
                <li>
                  <Link href="/gallery" className="hover:text-violet-400 transition-colors duration-300">
                    Gallery
                  </Link>
                </li>
                <li>
                  <Link href="/about" className="hover:text-violet-400 transition-colors duration-300">
                    About
                  </Link>
                </li>
              </ul>
            </nav>
            <button className="md:hidden text-2xl focus:outline-none hover:text-violet-400 transition-colors">
              <i className="fas fa-bars"></i>
            </button>
          </div>
        </header>

        {/* Main Content */}
        <main className="px-4 py-20">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-5xl font-bold text-center mb-12">
              Our <span className="gradient-text">Models</span>
            </h1>

            <section className="space-y-12">
              {/* Dataset */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Dataset</h2>
                <p className="text-gray-300 mb-2">
                  We used the open-source{" "}
                  <a
                    href="https://cocodataset.org/#home"
                    className="text-violet-400 underline hover:text-pink-400"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    COCO 2017 dataset
                  </a>{" "}
                  and further augmented it with 30k color-to-grayscale images from Flickr.
                </p>
                <p className="text-gray-300">
                  We performed an 80/10/10 split for train/validation/test. We ensured no data leakage using hash-based splitting.
                </p>
              </div>

              {/* Preprocessing */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Preprocessing</h2>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>Removed all grayscale images from the original dataset</li>
                  <li>Resized images to 256x256</li>
                  <li>Normalized all images in RGB</li>
                  <li>Converted all images from RGB to L*a*b</li>
                  <li>Normalized all L*a*b images</li>
                  <li>Used the L as the input for training and the *a*b as a ground truth</li>
                </ul>
              </div>

              {/* Models */}
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                {models.map((model) => (
                  <div key={model.name} className="glass-card p-6 rounded-3xl transition hover:-translate-y-2 hover-glow">
                    <h3 className="text-2xl font-bold mb-2 text-violet-300">{model.name}</h3>
                    <p className="text-gray-300 mb-4">{model.description}</p>
                    <h4 className="text-lg font-semibold text-purple-400">Hyperparameters</h4>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      {Object.entries(model.hyperparams).map(([key, value]) => (
                        <li key={key}>
                          <strong className="capitalize">{key}:</strong> {value}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>

              {/* Loss Functions */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Loss Functions</h2>
                <p className="text-gray-300 mb-2">
                  We used Mean Squared Error (MSE) for the baseline, and added perceptual loss from a pretrained VGG-19 on the U-Net model to improve perceptual similarity.
                </p>
                <p className="text-gray-300">
                  This helped guide the model toward more realistic, vivid outputs instead of simply minimizing pixel-wise differences.
                </p>
              </div>

              {/* Gallery Link */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Example Outputs</h2>
                <p className="text-gray-300 mb-4">
                  Click below to view real examples comparing grayscale input, ground truth, and our predictions.
                </p>
                <Link
                  href="/gallery"
                  className="inline-block bg-gradient-to-r from-pink-500 to-purple-500 px-6 py-3 rounded-xl text-white font-medium hover:opacity-90 transition"
                >
                  View Gallery
                </Link>
              </div>

              {/* Professionalism */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Professionalism</h2>
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li>Pre-commit hooks for linting and formatting</li>
                  <li>GitHub Actions for CI/CD</li>
                  <li>Containerized model using Docker</li>
                  <li>Branch protection rules and issue tracking</li>
                  <li>
                    Public dataset and model artifacts hosted on{" "}
                    <a
                      href="https://github.com/atodorov284/image_colorizer.git"
                      className="text-violet-400 underline hover:text-pink-400"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub
                    </a>
                  </li>
                </ul>
              </div>

              {/* Insights */}
              <div className="glass-card p-8 rounded-3xl">
                <h2 className="text-3xl font-semibold mb-4">Insights</h2>
                <p className="text-gray-300">
                  We observed overfitting in early experiments using shallow networks. The use of perceptual loss helped preserve texture and realism, while deeper networks performed better on detailed features like hair and clothing. We also discovered certain mislabels in the dataset, especially in crowd scenes, which affected learning stability.
                </p>
              </div>
            </section>
          </div>
        </main>

        {/* Footer */}
        <footer className="py-16 bg-slate-900/80 backdrop-blur-lg border-t border-white/10">
          <div className="container mx-auto px-4 text-center text-gray-500">
            <p>
              &copy; {new Date().getFullYear()} ChromaFlow. Made with{" "}
              <i className="fas fa-heart text-violet-400"></i> in Groningen.
            </p>
          </div>
        </footer>
      </div>
    </>
  );
};

export default ModelPage;