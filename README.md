# Onion-NN
PHP Deep Learning Framework

Experimental PHP based neural network with available deep learning constructs.
Just for fun.

# Why
A couple of deep learning enthusiast friends agreed on creating a NN implementation with a languange of choice just to proof and learn some important concepts regarding Deep Learning. I have a love-hate relation with PHP. I find it really bad as a language (specially in the past) but has been instrumental in my survival. So I decided to give some love back with a small NN framework given that, (afaik) there are not many options. Performance is definitely an issue but it might be of use for some corner cases or just as a learning aid.

I dediced to have a functional approach and also rely on arrays as structure but most likely will be moving to a mixed object (std object no custom classes) and SPL's fixed arrays as an optimization in the future.

# Dependencies
I have developed this testing against PHP v7 and you need php-stats module (gaussian distribution sampling).

# Running test
php -f test_file.php


