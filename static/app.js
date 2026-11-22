const toggle = document.querySelector(".nav-toggle");
const menu = document.getElementById("menu");

toggle.addEventListener("click", () => {
  const open = menu.classList.toggle("open");
  toggle.setAttribute("aria-expanded", String(open));
});

const form = document.querySelector("form.form");

if (form) {
  const samples = JSON.parse(form.dataset.samples);

  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      Object.entries(samples[button.dataset.sample]).forEach(([name, value]) => {
        form.elements[name].value = value;
      });
    });
  });

  form.addEventListener("submit", () => {
    const submit = form.querySelector("button[type=submit]");
    submit.textContent = "Menghitung...";
    submit.disabled = true;
  });
}
