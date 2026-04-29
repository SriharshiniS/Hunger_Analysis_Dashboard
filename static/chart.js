fetch('/api/chart-data')
.then(res => res.json())
.then(data => {
    new Chart(document.getElementById('chart'), {
        type: 'line',
        data: {
            labels: data.years,
            datasets: [{
                label: 'Undernourishment %',
                data: data.values
            }]
        }
    });
});