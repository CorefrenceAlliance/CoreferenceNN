package CoreferenceAlliance;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.w3c.dom.Document;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;


public class LSTM
{
   private MultiLayerNetwork net;

   private int is_candidate_anaphor_index = 0;
   private int is_candidate_antecedent_index = 1;
   private int anaphor_match_index = 2;
   private int antecedent_match_index = 3;

   private int pos_index_offset = 4;
   private String[] pos_indices = {"stuff"};


   private int input_size = pos_index_offset + pos_indices.length;
   private int first_layer_size = 100;
   private int second_layer_size = 100;
   private int output_size = 2;

   public void CreateNewNet()
   {
      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT )
            .learningRate( 0.1 )
            .rmsDecay( 0.95 )
            .seed( 259742343 )
            .regularization( true )
            .l2( 0.001 )
            .weightInit( WeightInit.XAVIER )
            .updater( Updater.RMSPROP )
            .list( 3 )
            .layer( 0, new GravesLSTM.Builder()
                  .nIn( input_size )
                  .nOut( first_layer_size )
                  .activation( "tanh" )
                  .build() )
            .layer( 1, new GravesLSTM.Builder()
                  .nIn( first_layer_size )
                  .nOut( second_layer_size )
                  .activation( "tanh" )
                  .build() )
            .layer( 2, new RnnOutputLayer.Builder()
                  .nIn( second_layer_size )
                  .nOut( output_size )
                  .activation( "softmax" )
                  .build() )
            .backpropType( BackpropType.Standard )
            .pretrain( false )
            .backprop( true )
            .build();

      net = new MultiLayerNetwork( conf );
      net.init();
   }

   public void TrainFromXMLDir( String XML_Dir )
   {
      List< AnnotatedDocument > docs = new ArrayList<>();

      // Number of times to run the net
      int total_inputs = 0;

      // Largest input sequence
      int max_input_length = 0;

      for( File xml : new File( XML_Dir ).listFiles() )
      {
         AnnotatedDocument doc = new AnnotatedDocument( xml );
         docs.add( doc );
         total_inputs += Math.pow( doc.mention_indices.size(), 2 ) - doc.mention_indices.size();
         max_input_length = Math.max( max_input_length, doc.lemmatized_tokens.size() + 1 );
      }


      INDArray features = Nd4j.zeros( total_inputs, input_size, max_input_length + 1 );
      INDArray labels = Nd4j.zeros( total_inputs, output_size, max_input_length + 1 );
      INDArray features_mask = Nd4j.zeros( total_inputs, max_input_length + 1 );
      INDArray labels_mask = Nd4j.zeros( total_inputs, max_input_length + 1 );

      int input_index = 0;
      for( AnnotatedDocument doc : docs )
      {
         INDArray pos_sequence = Nd4j.zeros( pos_indices.length, max_input_length );
         for( int i = 0; i < doc.pos_tags.size(); i++ )
         {
            int pos_index;
            for( pos_index = 0; pos_index < pos_indices.length; pos_index++ )
            {
               if( pos_indices[pos_index].equals( doc.pos_tags.get( i ) ) )
                  break;
            }
         }

         for( int i = 0; i < doc.mention_indices.size(); i++ )
         {
            for( int j = 0; j < doc.mention_indices.size(); j++ )
            {
               if( i == j )
                  continue;

               INDArray mention
            }
         }
      }
   }

   private class AnnotatedDocument
   {
      // This is horrible but quick and hopefully easyish to do

      // Grab all lemmatized tokens in the doc and stick them in here
      // The indices into this list will serve as identifiers for everything else
      public List< String > lemmatized_tokens;

      // The unique thing in the XML is the character offsets (can just take them directly as strings rather than parse them, I think)
      // This converts between those offsets and the indices
      public Map< String, Integer > offsets_to_indices;

      // The indices here are shared with lemmatized_tokens
      // IE pos_tags.get(1) is the tag for lemmatized_tokens.get(1)
      // Same for the mentions (Terms or TX's at the end of the file)
      public List< String > pos_tags;
      public List< Integer > mention_indices;

      // The indices between these two lists are linked
      // IE anaphor_indices.get(1) and antecedent_indices.get(1) form a cofreference (or Relation in the XML)
      // Again, the stored index refers back into lemmatized_tokens
      public List< Integer > anaphor_indices;
      public List< Integer > antecedent_indices;

      public AnnotatedDocument( File xml )
      {
         try
         {
            Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse( xml );

            int current_index = 0;


         }
         catch( Exception e )
         {
            e.printStackTrace();
         }

      }
   }

//   private class XMLAnnotationDirectoryIterator implements DataSetIterator
//   {
//      private class Token
//      {
//         private String text;
//         private String pos_tag;
//      }
//
//      List< File > files_remaining;
//      List< Token > tokens_current_file;
//      List< Integer > indices_mentions_remaining_current_file;
//      int batch_size;
//
//
//
//      public XMLAnnotationDirectoryIterator( String XML_Directory, int Batch_Size )
//      {
//         batch_size = Batch_Size;
//         for( File xml : new File( XML_Directory ).listFiles() )
//         {
//            files_remaining.add( xml );
//         }
//      }
//
//      public DataSet next()
//      {
//         return next(batch_size);
//      }
//
//      public DataSet next( int size )
//      {
//         int current_size = 0;
//
//         while( current_size < size && files_remaining.size() != 0 )
//         {
//
//         }
//
//         if( current_size != size ) throw NoSuchElementException;
//
//         DataSet result;
//         return result;
//      }
//   }
}
