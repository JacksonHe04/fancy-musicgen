import { useState, useEffect } from 'react'
import { Plus, X } from 'lucide-react'

const LeftSidebar = ({
  selectedScheme,
  setSelectedScheme,
  selectedModels,
  setSelectedModels,
  style,
  setStyle,
  tags,
  setTags
}) => {
  const [schemes, setSchemes] = useState([])
  const [tagInput, setTagInput] = useState('')

  // Fetch available schemes (output directories)
  useEffect(() => {
    const fetchSchemes = async () => {
      try {
        const response = await fetch('/api/schemes')
        const data = await response.json()
        setSchemes(data.schemes)
      } catch (error) {
        console.error('Failed to fetch schemes:', error)
        // Fallback: manually list output directories
        const fallbackSchemes = [
          'output',
          'output_epoch1',
          'output_epoch3',
          'output_fix',
          'output_fix2',
          'output_full'
        ]
        setSchemes(fallbackSchemes)
      }
    }
    fetchSchemes()
  }, [])

  // Update model options when scheme changes
  useEffect(() => {
    if (selectedScheme) {
      // Reset selected models when scheme changes
      setSelectedModels([])
    }
  }, [selectedScheme])

  const handleModelChange = (model) => {
    setSelectedModels(prev => {
      if (prev.includes(model)) {
        return prev.filter(m => m !== model)
      } else {
        return [...prev, model]
      }
    })
  }

  const handleAddTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()])
      setTagInput('')
    }
  }

  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove))
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddTag()
    }
  }

  const getAvailableModels = () => {
    const models = [
      { id: 'musicgen-small', name: 'MusicGen-Small', path: '/root/autodl-tmp/musicgen/local/musicgen-small' }
    ]
    
    if (selectedScheme) {
      models.push(
        { id: `${selectedScheme}-best`, name: `${selectedScheme} - Best Model`, path: `${selectedScheme}/best_model` },
        { id: `${selectedScheme}-final`, name: `${selectedScheme} - Final Model`, path: `${selectedScheme}/final_model` }
      )
    }
    
    return models
  }

  return (
    <div className="sidebar">
      <div className="form-group">
        <label className="form-label">微调方案</label>
        <select
          className="form-control"
          value={selectedScheme}
          onChange={(e) => setSelectedScheme(e.target.value)}
        >
          <option value="">请选择微调方案</option>
          {schemes.map(scheme => (
            <option key={scheme} value={scheme}>{scheme}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label className="form-label">模型选择</label>
        {getAvailableModels().map(model => (
          <div key={model.id} className="model-option">
            <input
              type="checkbox"
              id={model.id}
              checked={selectedModels.includes(model.id)}
              onChange={() => handleModelChange(model.id)}
              disabled={!selectedScheme && model.id !== 'musicgen-small'}
            />
            <label htmlFor={model.id}>{model.name}</label>
          </div>
        ))}
      </div>

      <div className="form-group">
        <label className="form-label">风格</label>
        <input
          type="text"
          className="form-control"
          placeholder="例如：欢快、悲伤、激昂..."
          value={style}
          onChange={(e) => setStyle(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label className="form-label">标签</label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            className="form-control"
            placeholder="添加标签"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleAddTag}
            disabled={!tagInput.trim()}
          >
            <Plus size={16} />
          </button>
        </div>
        <div className="tags-container">
          {tags.map(tag => (
            <span key={tag} className="tag">
              {tag}
              <button
                type="button"
                className="tag-remove"
                onClick={() => handleRemoveTag(tag)}
              >
                <X size={14} />
              </button>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default LeftSidebar