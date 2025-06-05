// globals.d.ts - Place this in your project root or types folder
declare global {
    interface Window {
      tailwind: {
        config: {
          theme: {
            extend: {
              colors: Record<string, string>;
              fontFamily: Record<string, string[]>;
            };
          };
        };
      };
    }
  }
  
  // Global color palette constants
  export const COLORS = {
    darkBlue: '#1d2d50',
    midBlue: '#3d5883',
    lightBlue: '#4d6a9a',
    accent: '#ff6b6b',
    gradientStart: '#3494E6',
    gradientEnd: '#EC6EAD'
  } as const;
  
  // Global configuration for ChromaFlow
  export const CHROMAFLOW_CONFIG = {
    maxFileSize: 10 * 1024 * 1024, // 10MB
    supportedFormats: ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'],
    processingDelay: 2500, // ms
    models: {
      histocolor: {
        name: 'HistoColor Pro',
        accuracy: 95,
        speed: 'medium'
      },
      landscape: {
        name: 'Landscape AI',
        accuracy: 92,
        speed: 'medium'
      },
      turbo: {
        name: 'TurboColor',
        accuracy: 85,
        speed: 'fast'
      }
    }
  } as const;
  
  // API types
  export interface ColorizeRequest {
    imageData: string;
    model: keyof typeof CHROMAFLOW_CONFIG.models;
    options?: {
      quality?: 'low' | 'medium' | 'high';
      preserveDetails?: boolean;
    };
  }
  
  export interface ColorizeResponse {
    success: boolean;
    colorizedImageUrl?: string;
    originalImageUrl?: string;
    processingTime?: number;
    error?: string;
  }
  
  // Custom CSS properties type
  declare module 'react' {
    interface CSSProperties {
      clipPath?: string;
    }
  }
  
  export {};