<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload CSV for Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            padding-top: 50px;
        }
        .container {
            max-width: 600px;
            background: #dde9ed; /* Fond blanc */
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            font-weight: 700;
            margin-bottom: 30px;
        }
        .btn-custom {
    background-color: #5f9ea0;
    color: white;
    border: none;
    padding: 10px 40px; /* Adjusted padding */
    border-radius: 5px;
    font-size: 16px;
    transition: background-color 0.3s ease;
}

        .btn-custom:hover {
            background-color: #5f9ea0;
        }
        .results {
            margin-top: 30px;
        }
        .table {
            margin-top: 20px;
        }
        th, td {
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Upload CSV for Prediction</h1>
        <form id="upload-form">
            <div class="form-group">
                <input type="file" class="form-control-file" id="file-input" name="file" required>
            </div>
            <div class="text-center"> <!-- Added this div -->
                <button type="submit" class="btn btn-custom">Predict</button>
            </div>
        </form>
        <div class="results">
            <table class="table table-striped" id="results-table" style="display: none;">
                <thead>
                    <tr>
                        <th>Prediction</th>
                        <th>Probabilities</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            $('#upload-form').on('submit', function(event) {
                event.preventDefault();
                var formData = new FormData();
                formData.append('file', $('#file-input')[0].files[0]);

                $.ajax({
                    url: '/predict',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(data) {
                        if (data.error) {
                            alert(data.error);
                        } else {
                            $('#results-table tbody').empty();
                            data.forEach(function(item) {
                                var row = '<tr><td>' + item.prediction + '</td><td>' + item.probabilities.join(', ') + '</td></tr>';
                                $('#results-table tbody').append(row);
                            });
                            $('#results-table').show();
                        }
                    },
                    error: function() {
                        alert('An error occurred while processing the file.');
                    }
                });
            });
        });
    </script>
</body>
</html>
