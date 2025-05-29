


import React, { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { Upload, MessageCircle, FileText, Activity, Users, TrendingUp, AlertTriangle } from "lucide-react";

function App() {
  const [msg, setMsg] = useState("");
  const [chat, setChat] = useState([]);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("connected"); // Mock as connected
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [stats, setStats] = useState({
    uploaded_files_count: 3,
    total_chunks: 127,
    total_queries: 45,
    avg_confidence: 0.82
  });
  const [activeTab, setActiveTab] = useState("chat");

  // Mock data for visualizations
  const [analyticsData] = useState({
    queryTrends: [
      { date: "2024-01-01", queries: 12, confidence: 0.78 },
      { date: "2024-01-02", queries: 18, confidence: 0.82 },
      { date: "2024-01-03", queries: 15, confidence: 0.79 },
      { date: "2024-01-04", queries: 22, confidence: 0.85 },
      { date: "2024-01-05", queries: 28, confidence: 0.88 },
      { date: "2024-01-06", queries: 25, confidence: 0.84 },
      { date: "2024-01-07", queries: 31, confidence: 0.89 }
    ],
    topicDistribution: [
      { name: "Cardiology", value: 35, color: "#8884d8" },
      { name: "Neurology", value: 28, color: "#82ca9d" },
      { name: "Oncology", value: 22, color: "#ffc658" },
      { name: "General Medicine", value: 15, color: "#ff7300" }
    ],
    confidenceDistribution: [
      { range: "0-20%", count: 2 },
      { range: "21-40%", count: 5 },
      { range: "41-60%", count: 8 },
      { range: "61-80%", count: 15 },
      { range: "81-100%", count: 25 }
    ]
  });

  // Initialize with mock data
  useEffect(() => {
    setUploadedFiles([
      {
        filename: "medical_textbook_cardiology.pdf",
        chunks_count: 45,
        file_size: 2048576,
        upload_date: "2024-01-15"
      },
      {
        filename: "patient_case_studies.txt", 
        chunks_count: 32,
        file_size: 1024000,
        upload_date: "2024-01-14"
      },
      {
        filename: "clinical_guidelines.pdf",
        chunks_count: 50,
        file_size: 3072000,
        upload_date: "2024-01-13"
      }
    ]);
  }, []);

  // Mock file upload
  const uploadFile = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    setUploadLoading(true);
    
    // Simulate upload delay
    setTimeout(() => {
      const newFile = {
        filename: file.name,
        chunks_count: Math.floor(Math.random() * 50) + 10,
        file_size: file.size,
        upload_date: new Date().toISOString().split('T')[0]
      };
      
      setUploadedFiles(prev => [...prev, newFile]);
      setStats(prev => ({
        ...prev,
        uploaded_files_count: prev.uploaded_files_count + 1,
        total_chunks: prev.total_chunks + newFile.chunks_count
      }));
      
      alert(`✅ File "${file.name}" uploaded successfully!`);
      setFile(null);
      document.querySelector('input[type="file"]').value = '';
      setUploadLoading(false);
    }, 2000);
  };

  // Mock chat responses
  const mockResponses = [
    {
      text: "Based on the uploaded medical literature, hypertension is typically defined as blood pressure readings consistently above 140/90 mmHg. Treatment often involves lifestyle modifications and antihypertensive medications.",
      confidence: 0.89,
      disclaimer: "This information is for educational purposes only. Always consult with healthcare professionals."
    },
    {
      text: "The symptoms you're describing could be related to several conditions. According to the clinical guidelines in your documents, it's important to consider patient history and perform proper diagnostic tests.",
      confidence: 0.76,
      disclaimer: "IMPORTANT: This AI cannot provide medical diagnosis. Please consult a qualified physician."
    },
    {
      text: "Cardiovascular disease prevention includes regular exercise, healthy diet, smoking cessation, and managing risk factors like diabetes and cholesterol levels as outlined in the cardiology textbook.",
      confidence: 0.92,
      disclaimer: "Follow evidence-based medical practices and professional guidance."
    }
  ];

  const sendMsg = async () => {
    if (!msg.trim()) return;

    setLoading(true);
    const userMessage = msg.trim();
    setMsg("");

    // Add user message to chat immediately
    setChat(prev => [...prev, { 
      user: userMessage, 
      bot: null,
      confidence: 0,
      disclaimer: ""
    }]);

    // Simulate API delay
    setTimeout(() => {
      const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      
      setChat(prev => {
        const newChat = [...prev];
        newChat[newChat.length - 1] = {
          user: userMessage,
          bot: randomResponse.text,
          confidence: randomResponse.confidence,
          disclaimer: randomResponse.disclaimer
        };
        return newChat;
      });
      
      // Update stats
      setStats(prev => ({
        ...prev,
        total_queries: prev.total_queries + 1,
        avg_confidence: (prev.avg_confidence + randomResponse.confidence) / 2
      }));
      
      setLoading(false);
    }, 1500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMsg();
    }
  };

  const clearChat = () => {
    setChat([]);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const TabButton = ({ id, label, icon: Icon, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
        isActive
          ? 'bg-blue-600 text-white'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      <Icon size={18} />
      <span>{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 text-blue-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center text-blue-800 mb-4">
            🩺 Medical Chat Assistant
          </h1>
          
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-blue-50 rounded-lg p-4 text-center">
              <FileText className="mx-auto mb-2 text-blue-600" size={24} />
              <div className="text-2xl font-bold text-blue-800">{stats.uploaded_files_count}</div>
              <div className="text-sm text-blue-600">Documents</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 text-center">
              <Activity className="mx-auto mb-2 text-green-600" size={24} />
              <div className="text-2xl font-bold text-green-800">{stats.total_chunks}</div>
              <div className="text-sm text-green-600">Text Chunks</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 text-center">
              <MessageCircle className="mx-auto mb-2 text-purple-600" size={24} />
              <div className="text-2xl font-bold text-purple-800">{stats.total_queries}</div>
              <div className="text-sm text-purple-600">Queries Processed</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4 text-center">
              <TrendingUp className="mx-auto mb-2 text-orange-600" size={24} />
              <div className="text-2xl font-bold text-orange-800">{(stats.avg_confidence * 100).toFixed(1)}%</div>
              <div className="text-sm text-orange-600">Avg Confidence</div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex flex-wrap gap-2 justify-center">
            <TabButton
              id="chat"
              label="Chat"
              icon={MessageCircle}
              isActive={activeTab === "chat"}
              onClick={setActiveTab}
            />
            <TabButton
              id="upload"
              label="Upload"
              icon={Upload}
              isActive={activeTab === "upload"}
              onClick={setActiveTab}
            />
            <TabButton
              id="analytics"
              label="Analytics"
              icon={Activity}
              isActive={activeTab === "analytics"}
              onClick={setActiveTab}
            />
            <TabButton
              id="files"
              label="Files"
              icon={FileText}
              isActive={activeTab === "files"}
              onClick={setActiveTab}
            />
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === "chat" && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-700">💬 Ask Questions</h2>
              {chat.length > 0 && (
                <button
                  onClick={clearChat}
                  className="text-sm text-gray-500 hover:text-gray-700 px-3 py-1 rounded border border-gray-300 hover:border-gray-400 transition-colors"
                >
                  🗑️ Clear Chat
                </button>
              )}
            </div>
            
            {/* Chat Input */}
            <div className="flex gap-3 mb-4">
              <input
                className="flex-grow border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={msg}
                onChange={e => setMsg(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about medical topics, treatments, or symptoms..."
                disabled={loading}
              />
              <button
                onClick={sendMsg}
                disabled={loading || !msg.trim()}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  loading || !msg.trim()
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                {loading ? '⏳' : '🚀 Ask'}
              </button>
            </div>

            {/* Chat History */}
            <div className="bg-gray-50 rounded-lg p-4 h-96 overflow-y-auto">
              {chat.length === 0 ? (
                <div className="text-center text-gray-500 mt-20">
                  <MessageCircle className="mx-auto mb-4 text-gray-400" size={48} />
                  <p className="text-lg">👋 Welcome!</p>
                  <p>Start asking questions about medical topics.</p>
                  <p className="text-sm mt-2">Try: "What are the symptoms of hypertension?" or "How is diabetes treated?"</p>
                </div>
              ) : (
                chat.map((entry, i) => (
                  <div key={i} className="mb-6 last:mb-0">
                    {/* User Message */}
                    <div className="flex justify-end mb-2">
                      <div className="bg-blue-500 text-white rounded-lg px-4 py-2 max-w-xs lg:max-w-md">
                        <p className="font-medium">You:</p>
                        <p>{entry.user}</p>
                      </div>
                    </div>
                    
                    {/* Bot Response */}
                    <div className="flex justify-start">
                      <div className="bg-white border rounded-lg px-4 py-2 max-w-xs lg:max-w-md shadow-sm">
                        <p className="font-medium text-blue-600 mb-1">🤖 Assistant:</p>
                        
                        {entry.bot === null ? (
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                            <span className="text-gray-500">Thinking...</span>
                          </div>
                        ) : (
                          <>
                            <p className="mb-2">{entry.bot}</p>
                            
                            {entry.confidence > 0 && (
                              <div className="text-xs text-gray-500 mb-1">
                                <div className="flex items-center space-x-2">
                                  <span>Confidence:</span>
                                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                                    <div 
                                      className={`h-2 rounded-full ${
                                        entry.confidence > 0.7 ? 'bg-green-500' :
                                        entry.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                                      }`}
                                      style={{ width: `${entry.confidence * 100}%` }}
                                    ></div>
                                  </div>
                                  <span>{(entry.confidence * 100).toFixed(1)}%</span>
                                </div>
                              </div>
                            )}
                            
                            {entry.disclaimer && (
                              <div className={`text-xs p-2 rounded border-l-4 ${
                                entry.disclaimer.includes('IMPORTANT') ? 
                                'bg-red-50 text-red-800 border-red-400' :
                                'bg-yellow-50 text-yellow-800 border-yellow-400'
                              }`}>
                                <AlertTriangle className="inline mr-1" size={12} />
                                {entry.disclaimer}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === "upload" && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">📄 Upload Medical Document</h2>
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <input
                type="file"
                accept=".pdf,.txt"
                onChange={(e) => setFile(e.target.files[0])}
                className="flex-grow p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={uploadFile}
                disabled={uploadLoading || !file}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  uploadLoading || !file
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {uploadLoading ? '⏳ Uploading...' : '📤 Upload Document'}
              </button>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              Supported formats: PDF, TXT (Max 10MB)
            </p>
          </div>
        )}

        {activeTab === "analytics" && (
          <div className="space-y-6">
            {/* Query Trends Chart */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">📈 Query Trends & Confidence</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analyticsData.queryTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Bar yAxisId="left" dataKey="queries" fill="#8884d8" name="Queries" />
                  <Line yAxisId="right" type="monotone" dataKey="confidence" stroke="#82ca9d" name="Avg Confidence" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Topic Distribution */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">🏥 Topic Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={analyticsData.topicDistribution}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {analyticsData.topicDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Confidence Distribution */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">🎯 Confidence Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={analyticsData.confidenceDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#ffc658" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {activeTab === "files" && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">📋 Uploaded Documents</h2>
            {uploadedFiles.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <FileText className="mx-auto mb-4 text-gray-400" size={48} />
                <p>No documents uploaded yet.</p>
                <p className="text-sm">Switch to the Upload tab to add documents.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-800 mb-1">{file.filename}</h3>
                        <div className="text-sm text-gray-500 space-y-1">
                          <div>📊 {file.chunks_count} text chunks processed</div>
                          <div>💾 File size: {formatFileSize(file.file_size)}</div>
                          <div>📅 Uploaded: {file.upload_date}</div>
                        </div>
                      </div>
                      <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                        ✅ Ready
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-6 text-gray-500 text-sm">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <AlertTriangle className="inline mr-2 text-yellow-600" size={16} />
            <span className="font-medium">Important Disclaimer:</span>
            <p className="mt-1">This is an AI assistant for informational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
