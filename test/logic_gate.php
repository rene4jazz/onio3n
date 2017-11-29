<?php

require_once(__DIR__ .'/../lib/model.php');
require_once(__DIR__ .'/../lib/layer/affine.php');
require_once(__DIR__ .'/../lib/layer/lrelu.php');
require_once(__DIR__ .'/../lib/trainer/online.php');

function run($nn, $input) {
  foreach ($input as $i => $X) {
    $Y = modelPredict($nn, $X);
    array_walk($Y, function(&$item){$item = round($item, 2);});
    print 'Prediction for:'."\n".
          'Input: '.implode(',', $X)."\n".
          'Output: '.implode(',', $Y)."\n";

  }
}

$input = [
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
];

$labels = [
  [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
  [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
  [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
  [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
];

$nn = [
  'hyper' => ['lr' => 0.02],
  'layers' => [
    // 2 inputs, 20 outputs (neurons) in layer
    [
      'type' => 'Affine',
      'weights' => initAffine(2, 20),
    ],
    ['type' => 'LReLU'],
    [ // 20 inputs, 6 outputs (neurons) in (last) layer
      'type' => 'Affine',
      'weights' => initAffine(20, 6),
    ],
    ['type' => 'LReLU'],
  ],
];

run($nn, $input);

onlineTrainer($nn, $input, $labels, 80000, function($e, $i){
  print $i.': '.$e."\n";
});
print "\n\n\n";
run($nn, $input);
