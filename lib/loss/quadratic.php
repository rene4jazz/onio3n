<?php

function quadratic($Y, $Y_hat, &$grad) { //Real vs Expected
  assert(count($Y) == count($Y_hat), 'Real vs Expected must have same dimension count');
  assert(is_array($grad), 'grad argument expected to be an array');

  $e = 0;
	foreach($Y as $i => $y) {
		$e += 0.5 * pow(((double)$Y_hat[$i] - (double)$y), 2);
    $grad[$i] = ((double)$Y_hat[$i] - (double)$y) * -1.0;
  }
	return $e;
}
