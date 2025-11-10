import express from 'express'
import cors from 'cors'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { dirname } from 'path'
import { spawn } from 'child_process'
import { promisify } from 'util'

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

// Python executable path (use conda environment)
// Try to use conda python first, fallback to system python
const PYTHON_PATH = process.env.PYTHON_PATH || (() => {
  // Try common conda python paths
  const possiblePaths = [
    '/root/miniconda3/envs/music/bin/python',
    '/opt/conda/envs/music/bin/python',
    'python3',
    'python'
  ]
  // For now, use python and let the system find it
  // User can set PYTHON_PATH environment variable if needed
  return 'python'
})()
const MODEL_SERVICE_PATH = path.join(__dirname, 'model_service.py')

// Helper function to run Python script
function runPythonScript(command, args) {
  return new Promise((resolve, reject) => {
    const pythonArgs = [MODEL_SERVICE_PATH, command, ...args]
    console.log(`Running: ${PYTHON_PATH} ${pythonArgs.join(' ')}`)
    
    const pythonProcess = spawn(PYTHON_PATH, pythonArgs, {
      cwd: __dirname,
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    })
    
    let stdout = ''
    let stderr = ''
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString()
      console.log(`Python stdout: ${data.toString()}`)
    })
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString()
      console.error(`Python stderr: ${data.toString()}`)
    })
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          // 只取最后一行为 JSON
          const lines = stdout.trim().split(/\r?\n/)
          const lastLine = lines[lines.length - 1]
          const result = JSON.parse(lastLine)
          resolve(result)
        } catch (e) {
          resolve(stdout.trim())
        }
      } else {
        reject(new Error(`Python script failed with code ${code}: ${stderr || stdout}`))
      }
    })
    
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`))
    })
  })
}

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
    // Filter out already loaded models
    const modelsToLoad = models.filter(modelId => !loadedModels.has(modelId))
    
    if (modelsToLoad.length === 0) {
      return res.json({ success: true, message: 'All models are already loaded' })
    }

    // Check if model paths exist
    for (const modelId of modelsToLoad) {
      let modelPath = ''
      if (modelId === 'musicgen-small') {
        modelPath = '/root/autodl-tmp/musicgen/local/musicgen-small'
      } else if (modelId.includes('-best')) {
        modelPath = `/root/autodl-tmp/musicgen/finetune/${scheme}/best_model`
      } else if (modelId.includes('-final')) {
        modelPath = `/root/autodl-tmp/musicgen/finetune/${scheme}/final_model`
      }

      if (modelPath && !fs.existsSync(modelPath)) {
        throw new Error(`Model path does not exist: ${modelPath}`)
      }
    }

    // Call Python service to load models
    console.log(`Loading models: ${modelsToLoad.join(', ')}`)
    const modelIdsJson = JSON.stringify(modelsToLoad)
    const results = await runPythonScript('load', [modelIdsJson, scheme || ''])
    
    // Check results and update loaded models
    let allSuccess = true
    const errors = []
    
    for (const modelId of modelsToLoad) {
      if (results[modelId] && results[modelId].success) {
        loadedModels.set(modelId, {
          path: modelId === 'musicgen-small' 
            ? '/root/autodl-tmp/musicgen/local/musicgen-small'
            : `/root/autodl-tmp/musicgen/finetune/${scheme}/${modelId.includes('-best') ? 'best' : 'final'}_model`,
          loadedAt: new Date().toISOString()
        })
        console.log(`Model ${modelId} loaded successfully`)
      } else {
        allSuccess = false
        const errorMsg = results[modelId]?.message || 'Unknown error'
        errors.push(`${modelId}: ${errorMsg}`)
        console.error(`Failed to load model ${modelId}: ${errorMsg}`)
      }
    }

    if (allSuccess) {
      res.json({ success: true, message: 'Models loaded successfully' })
    } else {
      res.status(500).json({ 
        success: false, 
        error: `Some models failed to load: ${errors.join('; ')}` 
      })
    }
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
    const unloadedModels = models.filter(modelId => !loadedModels.has(modelId))
    if (unloadedModels.length > 0) {
      throw new Error(`Models not loaded: ${unloadedModels.join(', ')}. Please load models first.`)
    }

    if (!style || !style.trim()) {
      throw new Error('Style is required')
    }

    const generatedFiles = []
    const outputDir = path.join(__dirname, 'public', 'generated')
    
    // Ensure the generated directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
    }
    
    // Generate music for each model
    for (const modelId of models) {
      console.log(`Generating music with model: ${modelId}`)
      
      try {
        // Call Python service to generate music
        const tagsJson = JSON.stringify(tags || [])
        const result = await runPythonScript('generate', [
          modelId,
          scheme || '',
          style.trim(),
          tagsJson,
          '10' // duration in seconds
        ])
        
        if (result.filename && result.filepath) {
          // Verify file exists
          if (fs.existsSync(result.filepath)) {
            generatedFiles.push({
              modelName: modelId,
              filename: result.filename,
              url: result.url || `/generated/${result.filename}`,
              style: style,
              tags: tags || [],
              scheme: scheme
            })
            console.log(`Music generated with model ${modelId}: ${result.filename}`)
          } else {
            throw new Error(`Generated file not found: ${result.filepath}`)
          }
        } else {
          throw new Error('Invalid response from model service')
        }
      } catch (error) {
        console.error(`Failed to generate music with model ${modelId}:`, error)
        // Continue with other models even if one fails
        generatedFiles.push({
          modelName: modelId,
          error: error.message,
          style: style,
          tags: tags || [],
          scheme: scheme
        })
      }
    }

    if (generatedFiles.length === 0) {
      throw new Error('Failed to generate music with any model')
    }

    res.json({ 
      success: true, 
      audioFiles: generatedFiles,
      message: `Music generated successfully with ${generatedFiles.filter(f => f.filename).length} model(s)`
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