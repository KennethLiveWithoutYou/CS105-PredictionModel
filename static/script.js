document.addEventListener("DOMContentLoaded", () => {

    // NAVIGATION
    const salaryBtn = document.getElementById("salary");
    const incomeBtn = document.getElementById("income");
    const salesBtn = document.getElementById("sales");

    if (salaryBtn) salaryBtn.addEventListener("click", () => window.location.href = "/salary");
    if (incomeBtn) incomeBtn.addEventListener("click", () => window.location.href = "/income");
    if (salesBtn) salesBtn.addEventListener("click", () => window.location.href = "/sales");

    // SALARY FORM
    const salaryForm = document.getElementById('salaryForm');
    const salaryResult = document.getElementById('result');

    if (salaryForm && salaryResult) {
        salaryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const years = parseFloat(document.getElementById('years').value);
            const response = await fetch('/predict_salary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ YearsExperience: years })
            });
            const data = await response.json();
            salaryResult.style.display = 'block';
            salaryResult.textContent = `Predicted Salary: ${data.predicted_salary.toFixed(2)}`;
        });
    }

    // INCOME FORM
    const incomeForm = document.getElementById('incomeForm');
    const incomeResult = document.getElementById('result');

    if (incomeForm && incomeResult) {
        incomeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const age = parseFloat(document.getElementById('age').value);
            const experience = parseFloat(document.getElementById('experience').value);
            const response = await fetch('/predict_income', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ age, experience })
            });
            const data = await response.json();
            incomeResult.style.display = 'block';
            incomeResult.textContent = `Predicted Income: ${data.predicted_income.toFixed(2)}`;
        });
    }

    // SALES FORM
    const salesForm = document.getElementById('salesForm');
    const salesResult = document.getElementById('result');

    if (salesForm && salesResult) {
        salesForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const TV = parseFloat(document.getElementById('TV').value);
            const Radio = parseFloat(document.getElementById('Radio').value);   
            const response = await fetch('/predict_sales', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ TV, Radio })
            });
            const data = await response.json();
            salesResult.style.display = 'block';
            salesResult.textContent = `Predicted Sales: ${data.predicted_sales.toFixed(2)}`;
        });
    }

});