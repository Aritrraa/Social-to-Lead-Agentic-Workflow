document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const resetBtn = document.getElementById('reset-btn');
    
    // Debug panel elements
    const debugIntent = document.getElementById('debug-intent');
    const debugStage = document.getElementById('debug-stage');
    const debugCaptured = document.getElementById('debug-captured');

    // Enable/disable send button based on input
    messageInput.addEventListener('input', () => {
        sendBtn.disabled = messageInput.value.trim() === '';
    });

    // Auto-resize input or handle enter key
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // 1. Add user message to UI
        appendMessage('user', message);
        messageInput.value = '';
        sendBtn.disabled = true;

        // 2. Show loading indicator
        const loadingId = showLoading();

        try {
            // 3. Send to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            
            // 4. Remove loading indicator
            removeMessage(loadingId);
            
            // 5. Add AI response to UI
            appendMessage('ai', formatAIResponse(data.response));
            
            // 6. Update debug panel
            updateDebugPanel(data.debug_info);

        } catch (error) {
            console.error('Error:', error);
            removeMessage(loadingId);
            appendMessage('ai', 'Sorry, I encountered an error. Please try again later.');
        }
    });

    // Reset chat
    resetBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: 'reset' })
            });
            
            // Clear messages
            chatMessages.innerHTML = '';
            appendMessage('ai', "Hello! I'm Alex from AutoStream. I've just been reset. How can I help you today?");
            
            // Reset debug panel
            updateDebugPanel({
                current_intent: 'Idle',
                lead_stage: 'None',
                lead_captured: false
            });
            
        } catch (error) {
            console.error('Reset error:', error);
        }
    });

    function appendMessage(sender, textOrHtml) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (sender === 'ai') {
            contentDiv.innerHTML = textOrHtml;
        } else {
            contentDiv.textContent = textOrHtml;
        }
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showLoading() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.id = id;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content loading-dots';
        
        contentDiv.innerHTML = `
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        `;
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) {
            el.remove();
        }
    }

    function updateDebugPanel(info) {
        if (!info) return;
        
        if (info.current_intent !== undefined) {
            debugIntent.textContent = info.current_intent || 'None';
        }
        
        if (info.lead_stage !== undefined) {
            debugStage.textContent = info.lead_stage || 'None';
        }
        
        if (info.lead_captured !== undefined) {
            debugCaptured.textContent = info.lead_captured ? 'True' : 'False';
            
            if (info.lead_captured) {
                debugCaptured.className = 'badge badge-success';
            } else {
                debugCaptured.className = 'badge badge-pending';
            }
        }
    }

    function formatAIResponse(text) {
        // Basic markdown formatting for the AI response
        if (!text) return '';
        
        let formatted = text
            // Replace newlines with <br>
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            // Bold text (**text**)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text (*text*)
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
        // Wrap in p tags if not already
        if (!formatted.startsWith('<p>')) {
            formatted = `<p>${formatted}</p>`;
        }
        
        return formatted;
    }
});
