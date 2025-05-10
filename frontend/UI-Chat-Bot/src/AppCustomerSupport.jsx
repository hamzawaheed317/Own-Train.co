import React, { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import { FiSend, FiUser, FiMessageSquare, FiX } from "react-icons/fi";

const AppCustomerSupport = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! How can I help you today?", sender: "bot" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: "user",
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue("");
    setIsTyping(true);

    // Simulate bot response after delay
    setTimeout(() => {
      const botResponses = [
        "I understand your concern. Let me check that for you.",
        "Thanks for your question. Our team will get back to you shortly.",
        "I've noted your request. Is there anything else I can help with?",
        "We appreciate your feedback. We'll work on improving this.",
      ];
      const randomResponse =
        botResponses[Math.floor(Math.random() * botResponses.length)];

      const newBotMessage = {
        id: messages.length + 2,
        text: randomResponse,
        sender: "bot",
      };
      setMessages((prev) => [...prev, newBotMessage]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <ChatContainer>
      {!isOpen ? (
        <ChatButton onClick={() => setIsOpen(true)}>
          <FiMessageSquare size={24} />
        </ChatButton>
      ) : (
        <ChatWindow>
          <ChatHeader>
            <h3>Customer Support</h3>
            <CloseButton onClick={() => setIsOpen(false)}>
              <FiX size={20} />
            </CloseButton>
          </ChatHeader>

          <MessagesContainer>
            {messages.map((message) => (
              <Message key={message.id} sender={message.sender}>
                {message.sender === "user" ? (
                  <UserIcon />
                ) : (
                  <BotIcon>B</BotIcon>
                )}
                <MessageText>{message.text}</MessageText>
              </Message>
            ))}
            {isTyping && (
              <Message sender="bot">
                <BotIcon>B</BotIcon>
                <TypingIndicator>
                  <span></span>
                  <span></span>
                  <span></span>
                </TypingIndicator>
              </Message>
            )}
            <div ref={messagesEndRef} />
          </MessagesContainer>

          <MessageForm onSubmit={handleSendMessage}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              aria-label="Type your message"
            />
            <SendButton type="submit" disabled={!inputValue.trim()}>
              <FiSend size={18} />
            </SendButton>
          </MessageForm>
        </ChatWindow>
      )}
    </ChatContainer>
  );
};

// Styled Components
const ChatContainer = styled.div`
  position: fixed;
  bottom: 30px;
  right: 30px;
  z-index: 1000;
`;

const ChatButton = styled.button`
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;

  &:hover {
    background: #1d4ed8;
    transform: scale(1.05);
  }
`;

const ChatWindow = styled.div`
  width: 350px;
  height: 500px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;

  @media (max-width: 480px) {
    width: 90vw;
    height: 70vh;
    bottom: 20px;
    right: 20px;
  }
`;

const ChatHeader = styled.div`
  background: #2563eb;
  color: white;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;

  h3 {
    margin: 0;
    font-size: 1.2rem;
  }
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  padding: 4px;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  background: #f9fafb;
`;

const Message = styled.div`
  display: flex;
  margin-bottom: 16px;
  justify-content: ${({ sender }) =>
    sender === "user" ? "flex-end" : "flex-start"};
`;

const MessageText = styled.div`
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 18px;
  background: ${({ sender }) => (sender === "user" ? "#2563eb" : "#e5e7eb")};
  color: ${({ sender }) => (sender === "user" ? "white" : "#111827")};
  margin-left: ${({ sender }) => (sender === "user" ? "0" : "8px")};
  margin-right: ${({ sender }) => (sender === "user" ? "8px" : "0")};
  word-break: break-word;
`;

const UserIcon = styled(FiUser)`
  width: 36px;
  height: 36px;
  background: #2563eb;
  color: white;
  border-radius: 50%;
  padding: 8px;
`;

const BotIcon = styled.div`
  width: 36px;
  height: 36px;
  background: #4b5563;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
`;

const TypingIndicator = styled.div`
  display: flex;
  align-items: center;
  padding: 10px 14px;
  background: #e5e7eb;
  border-radius: 18px;
  margin-left: 8px;

  span {
    height: 8px;
    width: 8px;
    background: #6b7280;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    animation: bounce 1.5s infinite ease-in-out;
  }

  span:nth-child(1) {
    animation-delay: 0s;
  }

  span:nth-child(2) {
    animation-delay: 0.2s;
  }

  span:nth-child(3) {
    animation-delay: 0.4s;
  }

  @keyframes bounce {
    0%,
    60%,
    100% {
      transform: translateY(0);
    }
    30% {
      transform: translateY(-5px);
    }
  }
`;

const MessageForm = styled.form`
  display: flex;
  padding: 12px;
  border-top: 1px solid #e5e7eb;

  input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    outline: none;
    font-size: 14px;

    &:focus {
      border-color: #2563eb;
    }
  }
`;

const SendButton = styled.button`
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  cursor: pointer;
  transition: background 0.3s ease;

  &:hover {
    background: #1d4ed8;
  }

  &:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }
`;

export default AppCustomerSupport;
