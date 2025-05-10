import React from "react";
import "./HomePage.css";
import { useState, useRef, useEffect } from "react";
import {
  PaperAirplaneIcon,
  UserIcon as UserSolid,
  ArrowPathIcon,
  TrashIcon,
  PlusIcon,
  Bars3Icon,
  XMarkIcon,
  MoonIcon,
} from "@heroicons/react/24/solid";
import { FaRobot } from "react-icons/fa";
import { format } from "date-fns";
import logo from "./final.png";
import AndroidIcon from "@mui/icons-material/Android";
import SmartToyIcon from "@mui/icons-material/SmartToy"; // Another good option
import FeedbackButtons from "./FeedbackWidget";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
  const [conversations, setConversations] = useState([
    { id: "1", title: "New conversation", messages: [] },
  ]);
  const [activeConversation, setActiveConversation] = useState("1");
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [theme, setTheme] = useState("light");
  const [model, setModel] = useState("gpt-4");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [rows, setRows] = useState(3);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const navigate = useNavigate();
  // Initialize with welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: "welcome",
          text: "Hello! I'm your AI assistant. How can I help you today?",
          sender: "bot",
          images: [],
          timestamp: new Date(),
        },
      ]);
    }
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      if (window.innerWidth > 768) {
        setShowSidebar(true);
      } else {
        setShowSidebar(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  async function handleUserQuery(userInput) {
    try {
      const user = JSON.parse(localStorage.getItem("user"));

      if (!user) {
        navigate("/login");
      }
      console.log("USer", user);
      const response = await fetch("http://localhost:5000/user/get-response", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userInput,
          prevHistory: messages,
          email: user.email,
          telephone: user.telephone,
        }),
      });

      console.log("response", response);

      if (!response.ok) throw new Error("API call failed");

      const data = await response.json();
      console.log("data", data);
      return {
        textResponse: data.response, // ٹیکسٹ جواب
        images: data.matchedImages, // تصاویر کی Array
      };
      // return data.response;
    } catch (error) {
      console.error("Frontend error:", error);
      // return "Sorry, I'm having trouble responding right now.";
      return {
        textResponse: "Sorry, I'm having trouble responding right now.",
        images: [],
      };
    }
  }

  const handleSendMessage = (e) => {
    e.preventDefault();

    if (!inputValue.trim()) return;

    const newUserMessage = {
      id: Date.now().toString(),
      text: inputValue.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(async () => {
      //handling the user query ok
      console.log("Input is : ", inputValue);
      console.log(`User Response Send to backend`);
      const response = await handleUserQuery(inputValue);
      console.log(`User request resolved ! with answer`, response);
      const newBotMessage = {
        id: Date.now().toString(),
        text: response.textResponse,
        images: response.images || [],
        sender: "bot",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, newBotMessage]);
      setIsTyping(false);

      // Update conversation title if it's the first user message
      if (messages.filter((m) => m.sender === "user").length === 0) {
        const newTitle =
          inputValue.length > 30
            ? `${inputValue.substring(0, 30)}...`
            : inputValue;
        updateConversationTitle(activeConversation, newTitle);
      }
    }, 1500 + Math.random() * 1000);
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  const updateConversationTitle = (id, title) => {
    setConversations((prev) =>
      prev.map((conv) => (conv.id === id ? { ...conv, title } : conv))
    );
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };
  return (
    <div className={`app-container ${theme}`}>
      <div className="main-content-home">
        {showDeleteConfirm && (
          <div className="delete-modal">
            <div className={`delete-modal-content ${theme}`}>
              <p>Are you sure you want to delete this conversation?</p>
              <div className="delete-actions">
                <button
                  className={`delete-cancel ${theme}`}
                  onClick={() => setShowDeleteConfirm(null)}
                >
                  Cancel
                </button>
                <button
                  className="delete-confirm"
                  onClick={() => deleteConversation(showDeleteConfirm)}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}

        <div className={`chat-container ${theme}`}>
          <div className="app-header">
            <div className="header-left">
              <div className="app-title" style={{ color: "green" }}>
                Own Train
              </div>
            </div>
            <button
              className={`theme-toggle ${theme}`}
              onClick={toggleTheme}
              aria-label="Toggle theme"
            >
              <MoonIcon className="theme-icon" />
            </button>
          </div>

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-header">
                <div className="header-content">
                  {message.sender === "bot" && (
                    <div className="bot-avatar">
                      {/* <FaRobot /> */}
                      {/* <SmartToyIcon /> */}
                      {/* <AndroidIcon /> */}
                      <img
                        src={logo}
                        alt="logo"
                        style={{
                          width: "50px",
                          // height: "45px",
                          marginLeft: "-14px",
                          position: "relative",
                          right: "4px",
                        }}
                      />
                    </div>
                  )}
                  <div className="message-time">
                    {format(new Date(message.timestamp), "h:mm a")}
                  </div>

                  {message.sender === "user" && (
                    <div className="user-avatar">
                      <UserSolid width={16} />
                    </div>
                  )}
                </div>
              </div>
              <div className={`message-content ${message.sender}`}>
                {message.text}
                {message.sender === "bot" &&
                  message.images &&
                  message.images.length > 0 && (
                    <div
                      className="bot-images"
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "10px",
                        marginTop: "10px",
                      }}
                    >
                      {message.images.map((img, index) => (
                        <div key={index}>
                          <img
                            src={`http://localhost:5000/uploads/${img.ImageName}`}
                            alt={img.ImageName}
                            style={{
                              width: "200px",
                              height: "auto",
                              borderRadius: "8px",
                              border: "1px solid #ddd",
                            }}
                          />
                          <p style={{ fontSize: "12px", textAlign: "left" }}>
                            Hello! How can I help you today?
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                <div>
                  {message.sender == "bot" && (
                    <FeedbackButtons responseId={message.id} />
                  )}
                </div>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="message bot">
              <div className="message-header">
                <div className="bot-avatar" style={{ color: "green" }}>
                  Own Train
                </div>
                <div className="message-time">generating...</div>
              </div>
              <div className="message-content bot">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className={`input-container ${theme}`}>
          <form
            onSubmit={handleSendMessage}
            className={`message-form ${theme}`}
          >
            <textarea
              className={`message-input ${theme}`}
              value={inputValue}
              ref={textareaRef}
              onChange={handleInputChange}
              placeholder="Message AI Assistant..."
              aria-label="Type your message"
              rows={rows}
            />
            <div className="action-buttons">
              <button
                type="submit"
                disabled={!inputValue.trim()}
                className="send-button"
              >
                <PaperAirplaneIcon width={18} />
              </button>
            </div>
          </form>
          <div className="disclaimer">
            AI Assistant may produce inaccurate information. Consider verifying
            important facts.
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
