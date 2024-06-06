<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
</head>
<body>
    <h1>Prediction Results</h1>
    <table border="1">
        <thead>
            <tr>
                <th>Prediction</th>
                <th>Probabilities</th>
            </tr>
        </thead>
        <tbody>
            {% for prediction, probability in results %}
            <tr>
                <td>{{ prediction }}</td>
                <td>{{ probability }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <a href="/">Upload another file</a>
</body>
</html>
