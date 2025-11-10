import { useState, useEffect } from 'react'
import './App.css'
import LeftSidebar from './components/LeftSidebar'
import RightContent from './components/RightContent'

function App() {
  const [selectedScheme, setSelectedScheme] = useState('')
  const [selectedModels, setSelectedModels] = useState([])
  const [style, setStyle] = useState('')
  const [tags, setTags] = useState([])
  const [loadingLogs, setLoadingLogs] = useState([])
  const [generatedAudio, setGeneratedAudio] = useState([])
  const [loadedModels, setLoadedModels] = useState({})

  return (
    <div className="app">
      <div className="app-header">
        <h1>MusicGen Fine-tuning Interface</h1>
      </div>
      <div className="app-content">
        <LeftSidebar
          selectedScheme={selectedScheme}
          setSelectedScheme={setSelectedScheme}
          selectedModels={selectedModels}
          setSelectedModels={setSelectedModels}
          style={style}
          setStyle={setStyle}
          tags={tags}
          setTags={setTags}
        />
        <RightContent
          loadingLogs={loadingLogs}
          generatedAudio={generatedAudio}
          selectedModels={selectedModels}
          selectedScheme={selectedScheme}
          style={style}
          tags={tags}
          setLoadingLogs={setLoadingLogs}
          setGeneratedAudio={setGeneratedAudio}
          loadedModels={loadedModels}
          setLoadedModels={setLoadedModels}
        />
      </div>
    </div>
  )
}

export default App
