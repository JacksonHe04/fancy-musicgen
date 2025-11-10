import express from 'express'
import cors from 'cors'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json())
app.use(express.static('public'))

// Store loaded models in memory
const loadedModels = new Map()

// API Routes

// Get available schemes (output directories)
app.get('/api/schemes', async (req, res) => {
  try {
    const finetunePath = '/root/autodl-tmp/musicgen/finetune'
    const items = fs.readdirSync(finetunePath)
    const schemes = items.filter(item => 
      item.startsWith('output') && 
      fs.statSync(path.join(finetunePath, item)).isDirectory()
    )
    res.json({ schemes })
  } catch (error) {
    console.error('Failed to fetch schemes:', error)
    res.status(500).json({ 
      error: 'Failed to fetch schemes',
      schemes: ['output', 'output_epoch1', 'output_epoch3', 'output_fix', 'output_fix2', 'output_full']
    })
  }
})

// Load models
app.post('/api/load-models', async (req, res) => {
  const { models, scheme } = req.body
  
  try {
    for (const modelId of models) {
      if (loadedModels.has(modelId)) {
        console.log(`Model ${modelId} already loaded`)
        continue
      }

      let modelPath = ''
      if (modelId === 'musicgen-small') {
        modelPath = '/root/autodl-tmp/musicgen/local/musicgen-small'
      } else if (modelId.includes('-best')) {
        modelPath = `/root/autodl-tmp/musicgen/finetune/${scheme}/best_model`
      } else if (modelId.includes('-final')) {
        modelPath = `/root/autodl-tmp/musicgen/finetune/${scheme}/final_model`
      }

      // Check if model path exists
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Model path does not exist: ${modelPath}`)
      }

      // Simulate model loading (in real implementation, this would load the actual model)
      console.log(`Loading model: ${modelId} from ${modelPath}`)
      
      // Here you would integrate with your actual model loading logic
      // For now, we'll simulate it with a delay
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      loadedModels.set(modelId, {
        path: modelPath,
        loadedAt: new Date().toISOString()
      })
      
      console.log(`Model ${modelId} loaded successfully`)
    }

    res.json({ success: true, message: 'Models loaded successfully' })
  } catch (error) {
    console.error('Failed to load models:', error)
    res.status(500).json({ 
      success: false, 
      error: error.message 
    })
  }
})

// Generate music
app.post('/api/generate', async (req, res) => {
  const { models, scheme, style, tags, loadedModels: clientLoadedModels } = req.body
  
  try {
    // Check if required models are loaded
    for (const modelId of models) {
      if (!loadedModels.has(modelId)) {
        throw new Error(`Model ${modelId} is not loaded. Please load models first.`)
      }
    }

    const generatedFiles = []
    
    // Generate music for each model
    for (const modelId of models) {
      console.log(`Generating music with model: ${modelId}`)
      
      // Simulate music generation (in real implementation, this would call your model)
      await new Promise(resolve => setTimeout(resolve, 3000))
      
      // Create a mock audio file for demonstration
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const filename = `${timestamp}_${modelId}_${style.replace(/\s+/g, '_')}.wav`
      const outputPath = path.join(__dirname, 'public', 'generated', filename)
      
      // Ensure the generated directory exists
      const generatedDir = path.dirname(outputPath)
      if (!fs.existsSync(generatedDir)) {
        fs.mkdirSync(generatedDir, { recursive: true })
      }
      
      // For demo purposes, copy an existing audio file
      const demoAudioPath = '/root/autodl-tmp/musicgen/output_music/20251110_020501_0_Cheerful_piano_music.wav'
      if (fs.existsSync(demoAudioPath)) {
        fs.copyFileSync(demoAudioPath, outputPath)
      }
      
      generatedFiles.push({
        modelName: modelId,
        filename: filename,
        url: `/generated/${filename}`,
        style: style,
        tags: tags,
        scheme: scheme
      })
      
      console.log(`Music generated with model ${modelId}: ${filename}`)
    }

    res.json({ 
      success: true, 
      audioFiles: generatedFiles,
      message: 'Music generated successfully'
    })
  } catch (error) {
    console.error('Failed to generate music:', error)
    res.status(500).json({ 
      success: false, 
      error: error.message 
    })
  }
})

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok',
    loadedModels: Array.from(loadedModels.keys()),
    timestamp: new Date().toISOString()
  })
})

// Create public/generated directory if it doesn't exist
const generatedDir = path.join(__dirname, 'public', 'generated')
if (!fs.existsSync(generatedDir)) {
  fs.mkdirSync(generatedDir, { recursive: true })
}

// Serve static files
app.use('/generated', express.static(generatedDir))

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err)
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' })
})

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
  console.log(`Available endpoints:`)
  console.log(`  GET  /api/schemes - Get available fine-tuning schemes`)
  console.log(`  POST /api/load-models - Load models`)
  console.log(`  POST /api/generate - Generate music`)
  console.log(`  GET  /api/health - Health check`)
})