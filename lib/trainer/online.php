<?php

require_once(__DIR__ .'/../loss/quadratic.php');

function onlineTrainer(
                        &$model,
                        $input,
                        $output_hat,
                        $num_epoch = 1,
                        $callback = null) {
  for ($ep = 0; $ep <= $num_epoch; $ep++) {
    $iteration = 1;
    foreach($input as $label => $X) {
      // forward pass
      $Y = $X;
      foreach ($model['layers'] as &$layer) {
        $f = 'forward'.$layer['type'];
        $Y = $f($Y, $layer, true);
      }

      // compute error
      $error_grad = []; //error grad from last layer;
      $error = quadratic($Y, $output_hat[$label], $error_grad);
      if (is_callable($callback))
        $callback($error, $iteration);

      // backprop
      krsort($model['layers'], true);
      $grad = $error_grad;
      foreach ($model['layers'] as &$layer) {
        $f = 'backward'.$layer['type'];
        $grad = $f($grad, $layer, $model['hyper']['lr']);
      }
      ksort($model['layers'], true);
      $iteration++;
    }
  }
}
