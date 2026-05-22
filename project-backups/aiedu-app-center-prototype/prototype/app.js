const sectionsRoot = document.querySelector("#sections");
let toastTimer;

function appIsDisabled(app) {
  const target = String(app.target || "").trim();
  return app.source === 1 && app.appType === 1 && !target;
}

function toneClass(tone) {
  const value = String(tone || "blue").toLowerCase();
  return `tone-${["blue", "green", "gold", "purple", "orange", "pink"].includes(value) ? value : "blue"}`;
}

function showToast(message) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    toast.setAttribute("role", "status");
    document.body.append(toast);
  }

  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("is-visible");
  }, 2200);
}

function createCard(app) {
  const disabled = appIsDisabled(app);
  const item = document.createElement("li");
  const button = document.createElement("button");
  const tag = document.createElement("span");
  const badge = document.createElement("span");
  const content = document.createElement("span");
  const title = document.createElement("span");
  const description = document.createElement("span");

  button.type = "button";
  button.className = `app-card ${toneClass(app.tone)}`;
  button.setAttribute("aria-disabled", String(disabled));

  if (disabled) {
    tag.className = "developing-tag";
    tag.textContent = "开发中";
    button.append(tag);
  }

  badge.className = "badge";
  badge.setAttribute("aria-hidden", "true");
  badge.textContent = app.initial || app.title?.slice(0, 1) || "A";

  content.className = "card-content";
  title.className = "card-title";
  description.className = "card-description";
  title.textContent = app.title || "未命名应用";
  description.textContent = app.description || "";

  content.append(title, description);
  button.append(badge, content);

  button.addEventListener("click", () => {
    if (disabled) {
      showToast(`${app.title} 正在开发中`);
      return;
    }

    showToast(`${app.title}：原型中保留点击反馈，正式版会打开 ${app.target}`);
  });

  item.append(button);
  return item;
}

function createSection(section) {
  const category = section.category || {};
  const wrapper = document.createElement("section");
  const heading = document.createElement("h2");
  const list = document.createElement("ul");

  wrapper.className = "section";
  heading.textContent = category.title || "未命名分组";
  list.className = "card-grid";

  for (const app of section.items || []) {
    list.append(createCard(app));
  }

  wrapper.append(heading, list);
  return wrapper;
}

async function loadSections() {
  const response = await fetch("../source/app-center-sections.json");
  const payload = await response.json();
  return Array.isArray(payload.data) ? payload.data : [];
}

loadSections()
  .then((sections) => {
    sectionsRoot.replaceChildren(...sections.map(createSection));
  })
  .catch(() => {
    sectionsRoot.innerHTML = "";
    const empty = document.createElement("section");
    empty.className = "section";
    empty.innerHTML = "<h2>应用中心</h2>";
    showToast("本地数据读取失败，请通过本地服务打开 prototype/index.html");
    sectionsRoot.append(empty);
  });
