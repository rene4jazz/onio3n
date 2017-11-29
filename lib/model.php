<?php

function modelPredict($model_state, $input) {
  // forward pass
  $Y = $input;
  foreach ($model_state['layers'] as $layer) {
    $f = 'forward'.$layer['type'];
    $Y = $f($Y, $layer);
  }
  return $Y;
}
