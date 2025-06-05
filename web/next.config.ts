/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['images.unsplash.com'],
    // Or use remotePatterns for more security (Next.js 12.3+):
    // remotePatterns: [
    //   {
    //     protocol: 'https',
    //     hostname: 'images.unsplash.com',
    //     port: '',
    //     pathname: '/**',
    //   },
    // ],
  },
}

export default nextConfig