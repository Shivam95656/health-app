const user_id = localStorage.getItem("user_id");

if (!user_id) {
  window.location.href = "login.html";
}

function showAlert(message, type = "error") {
  const el = document.getElementById("alert");
  el.textContent = message;
  el.className = `alert alert-${type} show`;
}

async function saveProfile() {
  const age     = document.getElementById("age").value;
  const sex     = document.getElementById("sex").value;
  const cp      = document.getElementById("cp").value;
  const trestbps = document.getElementById("bp").value;
  const chol    = document.getElementById("chol").value;
  const btn     = document.getElementById("saveBtn");

  if (!age || sex === "" || cp === "" || !trestbps || !chol) {
    return showAlert("Please fill in all fields.");
  }

  btn.textContent = "Saving...";
  btn.classList.add("btn-loading");

  const res  = await fetch("/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, age, sex, cp, trestbps, chol })
  });

  const data = await res.json();
  btn.textContent = "Save Profile";
  btn.classList.remove("btn-loading");

  if (data.error) {
    showAlert(data.error, "error");
  } else {
    showAlert(data.message, "success");
  }
}