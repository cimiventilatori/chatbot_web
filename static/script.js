document.getElementById("send-btn").onclick = async () => {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";

  const chatBox = document.getElementById("chat-box");
  chatBox.innerHTML += `<div class="user"><b>Tu:</b> ${message}</div>`;

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  });

  const data = await res.json();
  chatBox.innerHTML += `<div class="bot"><b>Bot:</b> ${data.response}</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
};
