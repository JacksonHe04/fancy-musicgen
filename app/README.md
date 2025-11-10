# MusicGen Fine-tuning Interface

A React-based web interface for MusicGen fine-tuning and music generation.

## Features

- **Dynamic Scheme Selection**: Automatically detects and lists available fine-tuning schemes from output directories
- **Model Management**: Supports loading MusicGen-Small and fine-tuned models (best_model and final_model)
- **Persistent Model Loading**: Models remain loaded between generation sessions to avoid repeated loading
- **Real-time Logs**: Shows model loading and music generation progress
- **Audio Playback**: Built-in audio player for generated music with download functionality
- **Tag System**: Add multiple tags to describe the desired music style

## Project Structure

```
app/
├── src/
│   ├── components/
│   │   ├── LeftSidebar.jsx    # User input controls
│   │   └── RightContent.jsx   # Model loading logs and audio players
│   ├── App.jsx                # Main application component
│   └── App.css                # Application styles
├── public/
│   └── generated/             # Generated audio files
├── server.js                  # Express.js backend API
└── package.json
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server (frontend + backend):
```bash
npm run dev:full
```

This will start:
- Frontend development server on http://localhost:5173
- Backend API server on http://localhost:3001

### Development

- Frontend only: `npm run dev`
- Backend only: `npm run server`
- Build for production: `npm run build`

## Usage

1. **Select Fine-tuning Scheme**: Choose from available output directories
2. **Choose Models**: Select MusicGen-Small and/or fine-tuned models
3. **Set Parameters**: 
   - Enter music style (e.g., "欢快", "悲伤", "激昂")
   - Add relevant tags
4. **Load Models**: Click "加载模型" to load selected models (first time only)
5. **Generate Music**: Click "生成音乐" to create audio files
6. **Play/Download**: Use the audio players to listen to and download generated music

## API Endpoints

- `GET /api/schemes` - Get available fine-tuning schemes
- `POST /api/load-models` - Load selected models
- `POST /api/generate` - Generate music with loaded models
- `GET /api/health` - Health check

## Model Integration

The application is designed to integrate with your existing MusicGen models:

- **MusicGen-Small**: Located at `/root/autodl-tmp/musicgen/local/musicgen-small`
- **Fine-tuned Models**: Located in respective output directories (best_model/final_model)

To integrate with your actual model loading and generation logic, modify the `server.js` file:

1. Replace the mock model loading in `/api/load-models` with your actual model loading code
2. Replace the mock music generation in `/api/generate` with your actual generation logic
3. Update the file paths and model configurations as needed

## Configuration

The application automatically detects available fine-tuning schemes by scanning the `/root/autodl-tmp/musicgen/finetune` directory for folders starting with "output".

## Styling

The interface uses a clean, modern design with:
- Responsive layout with sidebar and main content areas
- Dark theme for logs display
- Card-based audio player components
- Intuitive form controls and feedback

## Troubleshooting

- **Models not loading**: Check that model paths exist and are accessible
- **Audio not playing**: Ensure generated audio files are being created in the public/generated directory
- **API errors**: Check browser console and server logs for detailed error messages
