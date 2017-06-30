#https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

#https://github.com/felixgwu/img_classification_pk_pytorch/blob/master/models/densenet.py

class _DenseLayer(nn.Sequential):
    def __init__(self,num_input_features,growth_rate,bn_size,drop_rate):
        super(_DenseLayer,self).__init__()
        self.add_module('norm.1',nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1',nn.ReLU(inplace=True)),
        
        #If the bottle neck mode is set, apply feature reduction to limit the growth of features
        #Why should we expand the number of features by bn_size*growth?
        if bn_size>0:
            



class _DenseBlock(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
        super(_DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('denselayer%d' % (i+1),layer)



class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=4,block_config=(6, 12, 24, 16), compression=0.5,
             num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8,
             num_classes=10):
        super(DenseNet, self).__init__()
        
        #The first Convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1,num_init_features,kernal_size = 3,stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        #Did not add the pooling layer to preserve dimension
        
        #There are 4 DenseBlocks in the original ImageNet
        #The number of layers in each Densnet is adjustable
        
        
        num_features = num_init_features
        for i,num_layers in enumerate(block_config):
            Dense_block = _DenseBlock(num_layers = num_layers,num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            #Add name to the Denseblock
            self.features.add_module('denseblock%d' % (i + 1), Dense_block)
            
            
            #Increase the number of features by the growth rate times the number
            #of layers in each Denseblock
            num_features += num_layers * growth_rate
            
            #check whether the current block is the last block
            #Add a transition layer to all Denseblocks except the last
            if i != len(block_config):
                #Reduce the number of output features in the transition layer
                
                nOutChannels = int(math.floor(nChannels*reduction))
                
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=nOutChannels)
                self.features.add_module('transition%d' % (i + 1), trans)
                #change the number of features for the next Dense block
                num_features = nOutChannels
                
            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))

            # Linear layer
            self.classifier = nn.Linear(num_features, num_classes)
            
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out
        
        