<?php
require_once("../../include/initialize.php");
 if(!isset($_SESSION['ADMIN_USERID'])){
	redirect(web_root."admin/index.php");
}
$content='templates/home.php';
$view = (isset($_GET['page']) && $_GET['page'] != '') ? $_GET['page'] : '';
switch ($view) {
  case '1' :
        // $title="Home"; 
    // $content='home.php'; 
    if ($_SESSION['ADMIN_ROLE']=='Cashier') {
        # code...
      redirect('orders/');

    } 
    if ($_SESSION['ADMIN_ROLE']=='Administrator') {
        # code... 

      redirect('meals/');

    } 
    break;  
  default :
 
      $title="Prediction Attrition of Employees"; 
  
}

require_once ("../theme/templates.php");
?>