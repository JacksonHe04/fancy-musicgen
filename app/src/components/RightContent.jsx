import { useState } from 'react'
import { Play, Pause, Download } from 'lucide-react'

const RightContent = ({
  loadingLogs,
  generatedAudio,
  selectedModels,
  selectedScheme,
  style,
  tags,
  setLoadingLogs,
  setGeneratedAudio,
  loadedModels,
  setLoadedModels
}) => {
  const [isGenerating, setIsGenerating] = useState(false)
  const [isLoadingModels, setIsLoadingModels] = useState(false)

  const loadModels = async () => {
    if (selectedModels.length === 0) {
      alert('请至少选择一个模型')
      return
    }

    setIsLoadingModels(true)
    setLoadingLogs(prev => [...prev, '开始加载模型...'])

    try {
      const response = await fetch('/api/load-models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          models: selectedModels,
          scheme: selectedScheme
        })
      })

      const data = await response.json()
      
      if (data.success) {
        setLoadingLogs(prev => [...prev, '模型加载成功！'])
        // Update loaded models state
        const newLoadedModels = {}
        selectedModels.forEach(model => {
          newLoadedModels[model] = true
        })
        setLoadedModels(prev => ({ ...prev, ...newLoadedModels }))
      } else {
        setLoadingLogs(prev => [...prev, `模型加载失败: ${data.error}`])
      }
    } catch (error) {
      console.error('Failed to load models:', error)
      setLoadingLogs(prev => [...prev, `模型加载失败: ${error.message}`])
    } finally {
      setIsLoadingModels(false)
    }
  }

  const generateMusic = async () => {
    if (selectedModels.length === 0) {
      alert('请至少选择一个模型')
      return
    }

    if (!style.trim()) {
      alert('请输入音乐风格')
      return
    }

    setIsGenerating(true)
    setLoadingLogs(prev => [...prev, '开始生成音乐...'])

    try {
      // Check if models need to be loaded
      const modelsToLoad = selectedModels.filter(model => !loadedModels[model])
      if (modelsToLoad.length > 0) {
        setLoadingLogs(prev => [...prev, `需要加载模型: ${modelsToLoad.join(', ')}`])
        await loadModels()
      }

      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          models: selectedModels,
          scheme: selectedScheme,
          style: style,
          tags: tags,
          loadedModels: loadedModels
        })
      })

      const data = await response.json()
      
      if (data.success) {
        setLoadingLogs(prev => [...prev, '音乐生成成功！'])
        setGeneratedAudio(data.audioFiles)
      } else {
        setLoadingLogs(prev => [...prev, `音乐生成失败: ${data.error}`])
      }
    } catch (error) {
      console.error('Failed to generate music:', error)
      setLoadingLogs(prev => [...prev, `音乐生成失败: ${error.message}`])
    } finally {
      setIsGenerating(false)
    }
  }

  const formatLogTime = () => {
    return new Date().toLocaleTimeString('zh-CN')
  }

  return (
    <div className="main-content">
      <div style={{ marginBottom: '2rem' }}>
        <button
          className="btn btn-primary"
          onClick={generateMusic}
          disabled={isGenerating || isLoadingModels || selectedModels.length === 0 || !style.trim()}
          style={{ marginRight: '1rem' }}
        >
          {isGenerating ? '生成中...' : '生成音乐'}
          {(isGenerating || isLoadingModels) && <span className="loading"></span>}
        </button>
        
        <button
          className="btn btn-secondary"
          onClick={loadModels}
          disabled={isLoadingModels || selectedModels.length === 0}
        >
          {isLoadingModels ? '加载中...' : '加载模型'}
          {isLoadingModels && <span className="loading"></span>}
        </button>
      </div>

      <div className="form-group">
        <label className="form-label">模型加载日志</label>
        <div className="logs-container">
          {loadingLogs.length === 0 ? (
            <div className="log-entry">等待用户操作...</div>
          ) : (
            loadingLogs.map((log, index) => (
              <div key={index} className="log-entry">
                [{formatLogTime()}] {log}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">生成的音频</label>
        {generatedAudio.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            padding: '3rem', 
            color: '#666',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            border: '2px dashed #dee2e6'
          }}>
            <p>暂无生成的音频</p>
            <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
              请填写左侧表单并点击"生成音乐"按钮
            </p>
          </div>
        ) : (
          <div className="audio-grid">
            {generatedAudio.map((audio, index) => (
              <AudioCard key={index} audio={audio} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

const AudioCard = ({ audio }) => {
  const [isPlaying, setIsPlaying] = useState(false)

  const togglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  const downloadAudio = () => {
    const link = document.createElement('a')
    link.href = audio.url
    link.download = audio.filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="audio-card">
      <h3>{audio.modelName}</h3>
      <div style={{ marginBottom: '1rem' }}>
        <small style={{ color: '#666' }}>
          风格: {audio.style} | 标签: {audio.tags.join(', ')}
        </small>
      </div>
      <audio
        controls
        className="audio-player"
        src={audio.url}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          className="btn btn-secondary"
          onClick={togglePlay}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          {isPlaying ? '暂停' : '播放'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={downloadAudio}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Download size={16} />
          下载
        </button>
      </div>
    </div>
  )
}

export default RightContent