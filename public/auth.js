function showAlert(message, type = "error") {
  const el = document.getElementById("alert");
  if (!el) return;
  el.textContent = message;
  el.className = `alert alert-${type} show`;
}

function clearAlert() {
  const el = document.getElementById("alert");
  if (!el) return;
  el.className = "alert";
  el.textContent = "";
}

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function setButtonLoading(btn, isLoading) {
  if (!btn) return;

  if (!btn.dataset.defaultLabel) {
    btn.dataset.defaultLabel = btn.textContent;
  }

  if (isLoading) {
    btn.textContent = btn.dataset.loadingLabel || "Please wait...";
    btn.classList.add("btn-loading");
  } else {
    btn.textContent = btn.dataset.defaultLabel;
    btn.classList.remove("btn-loading");
  }
}

function getPasswordStrength(password) {
  if (!password) {
    return {
      label: "Add a password",
      width: "0%",
      help: "Use letters, numbers, and symbols for a stronger password."
    };
  }

  let score = 0;
  if (password.length >= 6) score += 1;
  if (password.length >= 10) score += 1;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score += 1;
  if (/\d/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;

  if (score <= 1) {
    return {
      label: "Weak",
      width: "24%",
      help: "Add more characters and include numbers or symbols."
    };
  }

  if (score <= 3) {
    return {
      label: "Fair",
      width: "56%",
      help: "Good start. Mixed case or symbols will strengthen it further."
    };
  }

  if (score === 4) {
    return {
      label: "Good",
      width: "78%",
      help: "This is solid. A little more length makes it even safer."
    };
  }

  return {
    label: "Strong",
    width: "100%",
    help: "Strong password. You're set for a safer account setup."
  };
}

function updatePasswordStrength() {
  const passwordInput = document.getElementById("password");
  const label = document.getElementById("passwordStrengthLabel");
  const fill = document.getElementById("passwordStrengthFill");
  const help = document.getElementById("passwordStrengthHelp");

  if (!passwordInput || !label || !fill || !help) return;

  const state = getPasswordStrength(passwordInput.value);
  label.textContent = state.label;
  fill.style.width = state.width;
  help.textContent = state.help;
}

function setupPasswordToggles() {
  document.querySelectorAll(".toggle-password").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.target);
      if (!input) return;

      input.type = input.type === "password" ? "text" : "password";
      btn.textContent = input.type === "password" ? "Show" : "Hide";
    });
  });
}

function setupFormEnhancements() {
  document.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", clearAlert);
  });

  const passwordInput = document.getElementById("password");
  if (passwordInput) {
    passwordInput.addEventListener("input", updatePasswordStrength);
    updatePasswordStrength();
  }

  setupPasswordToggles();
}

async function register() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const confirmPasswordEl = document.getElementById("confirmPassword");
  const confirmPassword = confirmPasswordEl ? confirmPasswordEl.value : "";
  const btn = document.getElementById("registerBtn");

  if (!email || !password || (confirmPasswordEl && !confirmPassword)) {
    return showAlert("Please fill in all required fields.");
  }

  if (!isValidEmail(email)) {
    return showAlert("Please enter a valid email address.");
  }

  if (password.length < 6) {
    return showAlert("Password must be at least 6 characters long.");
  }

  if (confirmPasswordEl && password !== confirmPassword) {
    return showAlert("Passwords do not match yet. Please recheck them.");
  }

  setButtonLoading(btn, true);

  try {
    const res  = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });

    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok || data.error) {
      showAlert(data.error || "Could not create your account right now.");
      return;
    }

    showAlert(data.message || "Registered successfully.", "success");
    setTimeout(() => window.location.href = "login.html", 1600);
  } catch (error) {
    showAlert("Network error. Please try again in a moment.", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}

async function login() {
  const email    = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const btn      = document.getElementById("loginBtn");
  

  if (!email || !password) return showAlert("Please fill in all fields.");

  setButtonLoading(btn, true);

  try {
    const res  = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });

    const data = await res.json();

    if (data.user) {
      localStorage.setItem("user_id", data.user.id);
      localStorage.setItem("user_email", email);
      window.location.href = "profile.html";
    } else {
      showAlert(data.error || "Unable to sign in.", "error");
    }
  } catch (error) {
    showAlert("Network error. Please try again.", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}

document.addEventListener("DOMContentLoaded", setupFormEnhancements);
