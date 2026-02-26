const state = {
  chats: [],
  currentChat: null,
  pending: false,
  sourcePanelsOpen: {},
};

const el = {
  chatList: document.getElementById("chat-list"),
  messages: document.getElementById("messages"),
  messageInput: document.getElementById("message-input"),
  sendBtn: document.getElementById("send-btn"),
  sendForm: document.getElementById("send-form"),
  newChatBtn: document.getElementById("new-chat-btn"),
  deleteChatBtn: document.getElementById("delete-chat-btn"),
  chatTitle: document.getElementById("chat-title"),
  chatUpdated: document.getElementById("chat-updated"),
  memoryKInput: document.getElementById("memory-k-input"),
  saveSettingsBtn: document.getElementById("save-settings-btn"),
  typingIndicator: document.getElementById("typing-indicator"),
  errorAlert: document.getElementById("error-alert"),
};

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const data = isJson ? await response.json() : null;
  if (!response.ok) {
    const detail = data?.detail || data?.error || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return data;
}

function showError(message) {
  el.errorAlert.textContent = message;
  el.errorAlert.classList.remove("d-none");
}

function clearError() {
  el.errorAlert.textContent = "";
  el.errorAlert.classList.add("d-none");
}

function formatDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function sourcePanelKey(chatId, msg, index) {
  return `${chatId}::${msg.ts || "no-ts"}::${index}`;
}

function setSourcePanelOpen(key, isOpen) {
  state.sourcePanelsOpen[key] = Boolean(isOpen);
}

function isSourcePanelOpen(key) {
  return Boolean(state.sourcePanelsOpen[key]);
}

function pathToFileUri(pathValue) {
  if (!pathValue) return "";
  if (pathValue.startsWith("file://")) return pathValue;
  if (pathValue.startsWith("/")) return encodeURI(`file://${pathValue}`);
  return encodeURI(pathValue);
}

function sourceFileProxyHref(pathValue) {
  return `/api/source/file?path=${encodeURIComponent(pathValue)}`;
}

function uriToWebHref(uriValue) {
  if (!uriValue) return "";
  if (uriValue.startsWith("file://")) {
    try {
      const url = new URL(uriValue);
      const pathValue = decodeURIComponent(url.pathname || "");
      return sourceFileProxyHref(pathValue);
    } catch (_err) {
      return uriValue;
    }
  }
  return uriValue;
}

function buildSourceLineElement(line) {
  const row = document.createElement("div");
  row.className = "sources-line";

  if (!line) {
    row.textContent = " ";
    return row;
  }

  const trimmed = line.trimStart();
  const indent = line.slice(0, line.length - trimmed.length);
  if (indent) {
    row.append(document.createTextNode(indent));
  }

  if (trimmed.startsWith("PATH:")) {
    const pathValue = trimmed.slice("PATH:".length).trim();
    row.append(document.createTextNode("PATH: "));
    const link = document.createElement("a");
    link.className = "sources-link";
    link.href = sourceFileProxyHref(pathValue);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = pathValue;
    row.append(link);
    return row;
  }

  if (trimmed.startsWith("URI:")) {
    const uriValue = trimmed.slice("URI:".length).trim();
    row.append(document.createTextNode("URI:  "));
    const link = document.createElement("a");
    link.className = "sources-link";
    link.href = uriToWebHref(uriValue);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = uriValue;
    row.append(link);
    return row;
  }

  row.append(document.createTextNode(trimmed));
  return row;
}

function buildSourcesPanel(text) {
  const panel = document.createElement("div");
  panel.className = "sources-panel";
  const lines = String(text || "").split("\n");
  for (const line of lines) {
    panel.appendChild(buildSourceLineElement(line));
  }
  return panel;
}

function renderChatList() {
  el.chatList.innerHTML = "";
  if (!state.chats.length) {
    const item = document.createElement("div");
    item.className = "text-muted small p-2";
    item.textContent = "No chats yet";
    el.chatList.appendChild(item);
    return;
  }

  for (const chat of state.chats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className =
      "list-group-item list-group-item-action" +
      (state.currentChat?.chat_id === chat.chat_id ? " active" : "");
    button.innerHTML = `
      <div class="fw-semibold text-truncate">${escapeHtml(chat.title)}</div>
      <div class="small opacity-75">${escapeHtml(formatDate(chat.updated_at))}</div>
    `;
    button.addEventListener("click", () => selectChat(chat.chat_id));
    el.chatList.appendChild(button);
  }
}

function renderMessages(lastSourcesText = "") {
  el.messages.innerHTML = "";
  const chat = state.currentChat;
  if (!chat) {
    el.messages.innerHTML = '<div class="empty-state">Create or select a chat to start.</div>';
    return;
  }

  if (!chat.messages.length) {
    el.messages.innerHTML = '<div class="empty-state">Start with your first message.</div>';
    return;
  }

  if (lastSourcesText && chat.messages.length) {
    const lastMsg = chat.messages[chat.messages.length - 1];
    if (lastMsg && lastMsg.role === "assistant") {
      lastMsg._sourcesText = lastSourcesText;
    }
  }

  chat.messages.forEach((msg, index) => {
    const wrap = document.createElement("div");
    wrap.className = `bubble-wrap ${msg.role === "user" ? "user" : "assistant"}`;

    const container = document.createElement("div");
    container.className = `bubble ${msg.role === "user" ? "user" : "assistant"}`;

    const content = document.createElement("div");
    content.innerHTML = escapeHtml(msg.content).replace(/\n/g, "<br>");
    container.appendChild(content);

    const meta = document.createElement("div");
    meta.className = "meta mt-1";
    meta.textContent = `${msg.role} • ${formatDate(msg.ts)}`;
    container.appendChild(meta);

    if (msg.role === "assistant" && typeof msg._sourcesText === "string" && msg._sourcesText.trim()) {
      const panelKey = sourcePanelKey(chat.chat_id, msg, index);
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "btn btn-link btn-sm p-0 sources-toggle";
      const open = isSourcePanelOpen(panelKey);
      toggle.textContent = open ? "Hide sources" : "Sources";

      const panelWrap = document.createElement("div");
      panelWrap.className = "sources-box";
      panelWrap.appendChild(toggle);

      const panel = buildSourcesPanel(msg._sourcesText);
      panel.hidden = !open;
      panelWrap.appendChild(panel);

      toggle.addEventListener("click", () => {
        const nextOpen = panel.hidden;
        panel.hidden = !nextOpen;
        setSourcePanelOpen(panelKey, nextOpen);
        toggle.textContent = nextOpen ? "Hide sources" : "Sources";
      });

      container.appendChild(panelWrap);
    }

    wrap.appendChild(container);
    el.messages.appendChild(wrap);
  });

  el.messages.scrollTop = el.messages.scrollHeight;
}

function syncChatHeader() {
  const chat = state.currentChat;
  if (!chat) {
    el.chatTitle.textContent = "No chat selected";
    el.chatUpdated.textContent = "";
    el.memoryKInput.value = 6;
    return;
  }
  el.chatTitle.textContent = chat.title;
  el.chatUpdated.textContent = `Updated: ${formatDate(chat.updated_at)}`;
  el.memoryKInput.value = chat.settings?.memory_k ?? 6;
}

function setPending(flag) {
  state.pending = flag;
  el.messageInput.disabled = flag || !state.currentChat;
  el.sendBtn.disabled = flag || !state.currentChat;
  el.newChatBtn.disabled = flag;
  el.deleteChatBtn.disabled = flag || !state.currentChat;
  el.saveSettingsBtn.disabled = flag || !state.currentChat;
  el.memoryKInput.disabled = flag || !state.currentChat;
  el.typingIndicator.classList.toggle("show", flag);
}

async function refreshChats() {
  state.chats = await apiFetch("/api/chats");
  renderChatList();
}

async function selectChat(chatId) {
  clearError();
  const chat = await apiFetch(`/api/chats/${encodeURIComponent(chatId)}`);
  state.currentChat = chat;
  renderChatList();
  syncChatHeader();
  renderMessages();
  setPending(state.pending);
}

async function createNewChat() {
  clearError();
  const chat = await apiFetch("/api/chats", { method: "POST", body: JSON.stringify({}) });
  await refreshChats();
  state.currentChat = chat;
  renderChatList();
  syncChatHeader();
  renderMessages();
  setPending(false);
  el.messageInput.focus();
}

async function deleteCurrentChat() {
  const chat = state.currentChat;
  if (!chat) return;
  const ok = window.confirm(`Delete chat "${chat.title}"?`);
  if (!ok) return;
  clearError();
  await apiFetch(`/api/chats/${encodeURIComponent(chat.chat_id)}`, { method: "DELETE" });
  const deletedId = chat.chat_id;
  await refreshChats();
  if (state.currentChat?.chat_id === deletedId) {
    state.currentChat = null;
  }
  if (state.chats.length) {
    await selectChat(state.chats[0].chat_id);
  } else {
    state.currentChat = null;
    renderChatList();
    syncChatHeader();
    renderMessages();
    setPending(false);
  }
}

async function saveSettings() {
  const chat = state.currentChat;
  if (!chat) return;
  clearError();
  const memoryK = Number(el.memoryKInput.value);
  if (!Number.isInteger(memoryK) || memoryK < 0 || memoryK > 100) {
    showError("memory_k must be an integer between 0 and 100");
    return;
  }
  const updated = await apiFetch(`/api/chats/${encodeURIComponent(chat.chat_id)}/settings`, {
    method: "PATCH",
    body: JSON.stringify({ memory_k: memoryK }),
  });
  state.currentChat = updated;
  await refreshChats();
  renderChatList();
  syncChatHeader();
  renderMessages();
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.pending || !state.currentChat) return;
  const text = el.messageInput.value.trim();
  if (!text) return;

  clearError();
  setPending(true);
  const chatId = state.currentChat.chat_id;
  el.messageInput.value = "";

  try {
    const result = await apiFetch("/api/chat/send", {
      method: "POST",
      body: JSON.stringify({ chat_id: chatId, message: text }),
    });
    state.currentChat = result.chat;
    if (state.currentChat?.messages?.length) {
      const lastMsg = state.currentChat.messages[state.currentChat.messages.length - 1];
      if (lastMsg?.role === "assistant") {
        lastMsg._sourcesText = result.sources_text || "";
      }
    }
    await refreshChats();
    renderChatList();
    syncChatHeader();
    renderMessages(result.sources_text || "");
  } catch (err) {
    showError(err.message || "Failed to send message");
    await selectChat(chatId);
  } finally {
    setPending(false);
    el.messageInput.focus();
  }
}

async function boot() {
  setPending(false);
  clearError();
  try {
    await refreshChats();
    if (state.chats.length) {
      await selectChat(state.chats[0].chat_id);
    } else {
      await createNewChat();
    }
  } catch (err) {
    showError(err.message || "Failed to load chats");
  }
}

el.sendForm.addEventListener("submit", sendMessage);
el.newChatBtn.addEventListener("click", () => {
  if (!state.pending) {
    createNewChat().catch((err) => showError(err.message || "Failed to create chat"));
  }
});
el.deleteChatBtn.addEventListener("click", () => {
  if (!state.pending) {
    deleteCurrentChat().catch((err) => showError(err.message || "Failed to delete chat"));
  }
});
el.saveSettingsBtn.addEventListener("click", () => {
  if (!state.pending) {
    saveSettings().catch((err) => showError(err.message || "Failed to save settings"));
  }
});

boot();
