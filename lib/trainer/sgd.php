<?php

require_once(__DIR__ .'/../loss/mse.php');

function batchTrainer($model, $input, $output_hat) {
  $output = [];
  foreach($input as $label => $X) {
    // forward pass
    $Y = $X;
    foreach ($model['layers'] as $index => &$layer) {
      $f = 'forward'.$layer['type'];
      $Y = (key_exists('weights', $layer)) ? $f($Y, $layer['weights'], true) : $f($Y, true);
    }
    $output[] = $Y;
  }

  // compute cost
  $e = mse($output, $output_hat);

  // final Grad
  $grad = [];

  print_r($e);
}
