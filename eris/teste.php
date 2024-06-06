<?php
// URL pour récupérer les résultats de matching depuis Flask
$url = 'http://localhost:5000/match';

// Effectuer une requête GET pour récupérer les résultats de matching
$response = file_get_contents($url);

// Décoder les données JSON
$matching_results = json_decode($response, true);

// Initialiser les tableaux pour les données du graphique
$cv_job_ids = [];
$similarities = [];

// Vérifier si des résultats sont disponibles
if ($matching_results) {
    // Parcourir les résultats de matching pour récupérer les données nécessaires pour le graphique
    foreach ($matching_results as $result) {
        $cv_job_ids[] = $result['cv_job_id'];
        $similarities[] = $result['similarity'];
    }
}

// Convertir les données du graphique en format JSON
$graph_data_json = json_encode(['cv_job_ids' => $cv_job_ids, 'similarities' => $similarities]);
?>



<!doctype html>
<html>
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Similarité entre CV et Offres d'emploi</title>
    <!-- Inclure la bibliothèque Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>

    <body>
        

        <?php

        if($_SERVER['REQUEST_METHOD'] === 'POST'){
            $username = $_POST['username'];
            $password = $_POST['password'];

            $response = file_get_contents('http://localhost:5000/validate?username='.urldecode($username). '&password=' .urlencode ($password));
            
            echo $response;


        }
        ?>
        
        <div id="graph"></div>
        <canvas id="myChart" width="400" height="400"></canvas>
    
        <h1>Similarité entre CV et Offres d'emploi</h1>

    <div id="graph"></div>

    <script>
        // Récupérer les données JSON générées par PHP
        var graphData = <?php echo $graph_data_json; ?>;

        // Créer le graphique interactif avec Plotly.js
        var traces = [];
        for (var i = 0; i < graphData.cv_job_ids.length; i++) {
            var trace = {
                x: [graphData.similarities[i]],
                y: [graphData.cv_job_ids[i]],
                name: graphData.cv_job_ids[i],
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: 'rgba(' + Math.floor(Math.random() * 256) + ', ' + Math.floor(Math.random() * 256) + ', ' + Math.floor(Math.random() * 256) + ', 0.6)' // Couleur aléatoire pour chaque CV
                }
            };
            traces.push(trace);
        }
        var layout = {
            title: 'Similarité entre CV et Offres d\'emploi',
            yaxis: { title: 'CV - Offre d\'emploi', automargin: true },
            xaxis: { title: 'Similarité' },
            barmode: 'stack',
            margin: { t: 50 }
        };
        Plotly.newPlot('graph', traces, layout);
    </script>
    </body>
</html>