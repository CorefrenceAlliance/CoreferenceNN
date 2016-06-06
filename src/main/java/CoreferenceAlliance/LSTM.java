package CoreferenceAlliance;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.io.XMLEntityReader;
import gov.nih.nlm.ling.io.XMLImplicitRelationReader;
import gov.nih.nlm.ling.io.XMLReader;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.ImplicitRelation;
import gov.nih.nlm.ling.sem.SemanticItem;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.File;
import java.util.*;
import java.util.Random;


public class LSTM
{
   private MultiLayerConfiguration conf;
   private MultiLayerNetwork net;

   private int do_output_index = 0;
   private int is_candidate_anaphor_index = 1;
   private int is_candidate_antecedent_index = 2;
   private int anaphor_match_index = 3;
   private int antecedent_match_index = 4;

   private int pos_index_offset = 5;
   private enum pos { CC, CD, DT, EX, FW, IN, JJ, JJR, JJS, LS, MD, NN, NNP,
         NNPS, NNS, PDT, POS, PRP, RB, RBR, RP, SYM, TO, UH, VB, VBD, VBN, VBP, VBZ,
         WDT, WP, WRB, LRB, RRB, VBG, RBS, PERIOD, COMMA, COLON, DACCENT, DSQUOTE }

   private int input_size = pos_index_offset + pos.values().length;
   private int first_layer_size = 100;
   private int second_layer_size = 100        ;
   private int output_size = 2;

   private int output_yes_index = 0;
   private int output_no_index = 1;

   Random random = new Random();


   public void CreateNewNet()
   {
      conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT )
            .iterations( 1 )
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
                  .lossFunction( LossFunctions.LossFunction.MCXENT )
                  .activation( "softmax" )
                  .build() )
            .backpropType( BackpropType.Standard )
            .pretrain( false )
            .backprop( true )
            .build();

      net = new MultiLayerNetwork( conf );
      net.init();
   }

   private Document XMLToDoc( File xml )
   {
      Map< Class< ? extends SemanticItem >, List< String > > annotationTypes = new HashMap<>();
      annotationTypes.put( Entity.class, Arrays.asList( "Drug", "Drug_Class", "Substance", "SPAN",
            "DefiniteNP", "IndefiniteNP", "ZeroArticleNP", "DemonstrativeNP", "DistributiveNP",
            "PersonalPronoun", "PossessivePronoun", "DemonstrativePronoun", "DistributivePronoun",
            "ReciprocalPronoun", "RelativePronoun", "IndefinitePronoun" ) );
      annotationTypes.put( ImplicitRelation.class, Arrays.asList( "Anaphora", "Cataphora", "PredicateNominative", "Appositive" ) );
      XMLReader reader = new XMLReader();
      reader.addAnnotationReader( ImplicitRelation.class, new XMLImplicitRelationReader() );
      reader.addAnnotationReader( Entity.class, new XMLEntityReader() );
      Document doc = reader.load( xml.getAbsolutePath(), null, null, true, null, annotationTypes, null );
      return doc;
   }

   private DataSet XMLDirToDatSet( String XML_Dir )
   {
      List< Document > docs = new ArrayList<>();

      int total_inputs = 0;
      int input_length = 0;

      for( File xml : new File( XML_Dir ).listFiles() )
      {
         Document doc = XMLToDoc( xml );

         docs.add( doc );
         input_length = Math.max( input_length, doc.getAllSurfaceElements().size() );

         LinkedHashSet< SemanticItem > relations = getAnaphoraRelations( doc );

         total_inputs += relations.size() * 2;     // Will generate as many negative cases as found positives
      }
      input_length++;   //final output step

//      System.out.println( total_inputs + " - " + input_length );

      INDArray features = Nd4j.zeros( total_inputs, input_size, input_length );
      INDArray labels = Nd4j.zeros( total_inputs, output_size, input_length );
      INDArray features_mask = Nd4j.zeros( total_inputs, input_length );
      INDArray labels_mask = Nd4j.zeros( total_inputs, input_length );

      int current_input = 0;

      for( int doc_num = 0; doc_num < docs.size(); doc_num++ )
      {
         Document doc = docs.get( doc_num );
         List< SurfaceElement > words = doc.getAllSurfaceElements();
         LinkedHashSet< SemanticItem > terms = getTerms( doc );
         LinkedHashSet< SemanticItem > relations = getAnaphoraRelations( doc );

         Map< SurfaceElement, Set< SurfaceElement > > used_relations = new HashMap<>();

         int relation_num = 0;
         // Positives
         for( SemanticItem relation : relations )
         {
            List< SurfaceElement > involved_words = doc.getSurfaceElementsInSpan( relation.getSpan() );
            SurfaceElement anaphor;
            SurfaceElement antecedent;
            if( SpanList.atLeft( involved_words.get( 0 ).getSpan(), involved_words.get( 1 ).getSpan() ) )
            {
               antecedent = involved_words.get( 0 );
               anaphor = involved_words.get( 1 );
            }
            else
            {
               anaphor = involved_words.get( 0 );
               antecedent = involved_words.get( 1 );
            }

            used_relations.putIfAbsent( antecedent, new HashSet<>() );
            used_relations.get( antecedent ).add( anaphor );

            // Set the output bit after the last word
//            features.put( new INDArrayIndex[] {
//                  NDArrayIndex.point( ( doc_num + 1 ) * relation_num ),
//                  NDArrayIndex.point( do_output_index ),
//                  NDArrayIndex.point( words.size() ) }, 1.0 );

//            System.out.println( current_input );
            features.putScalar( new int[] {
                  current_input,
                  do_output_index,
                  words.size() }, 1.0 );

            for( int word_num = 0; word_num < words.size(); word_num++ )
            {
               SurfaceElement word = words.get( word_num );
               boolean candidate_anaphor = anaphor.getSpan().equals( word.getSpan() );
               boolean candidate_antecedent = antecedent.getSpan().equals( word.getSpan() );
               boolean match_lemma_anaphor = anaphor.getLemma().equals( word.getLemma() );
               boolean match_lemma_antecdent = antecedent.getLemma().equals( word.getLemma() );

               String word_pos = cleanupPOS( word.getPos() );

               int pos_index;
               try
               {
                  pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();
               } catch( Exception e )
               {
                  System.out.println( word.getPos() );
                  System.out.println( "ADD THIS" );
                  throw e;
               }

               if( candidate_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_anaphor_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( candidate_antecedent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_antecedent_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( anaphor_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_antecdent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( antecedent_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               features.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( pos_index ),
                     NDArrayIndex.point( word_num ) }, 1.0 );

               features_mask.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( word_num ) }, 1.0 );
            }

            labels.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( output_yes_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );


            labels_mask.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            relation_num++;
            current_input++;
         }
         // Negatives
         for( ; relation_num < relations.size() * 2; relation_num++ )
         {
            SurfaceElement bad_anaphor;
            SurfaceElement bad_antecedent;
            do
            {
               int bad_anaphor_index = random.nextInt( terms.size() );
               int bad_antecedent_index = random.nextInt( terms.size() );

               Iterator< SemanticItem > iter = terms.iterator();
               for( int i = 0; i < bad_anaphor_index - 1; i++ )
                  iter.next();

               bad_anaphor = doc.getSurfaceElementsInSpan( iter.next().getSpan() ).get( 0 );

               iter = terms.iterator();
               for( int i = 0; i < bad_antecedent_index - 1; i++ )
                  iter.next();

               bad_antecedent = doc.getSurfaceElementsInSpan( iter.next().getSpan() ).get( 0 );
            }
            while(
                  used_relations.containsKey( bad_antecedent ) &&
                        used_relations.get( bad_antecedent ).contains( bad_anaphor ) );

            used_relations.putIfAbsent( bad_antecedent, new HashSet<>() );
            used_relations.get( bad_antecedent ).add( bad_anaphor );

            // Set the output bit after the last word
            features.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( do_output_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            features_mask.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            for( int word_num = 0; word_num < words.size(); word_num++ )
            {
               SurfaceElement word = words.get( word_num );
               boolean candidate_anaphor = bad_anaphor.getSpan().equals( word.getSpan() );
               boolean candidate_antecedent = bad_antecedent.getSpan().equals( word.getSpan() );
               boolean match_lemma_anaphor = bad_anaphor.getLemma().equals( word.getLemma() );
               boolean match_lemma_antecdent = bad_antecedent.getLemma().equals( word.getLemma() );

               String word_pos = cleanupPOS( word.getPos() );

               int pos_index;
               try
               {
                  pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();
               } catch( Exception e )
               {
                  System.out.println( word.getPos() );
                  System.out.println( "ADD THIS" );
                  throw e;
               }

//               int pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();

               if( candidate_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_anaphor_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( candidate_antecedent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_antecedent_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( anaphor_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_antecdent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( antecedent_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               features.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( pos_index ),
                     NDArrayIndex.point( word_num ) }, 1.0 );

               features_mask.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( word_num ) }, 1.0 );
            }

            labels.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( output_no_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            labels_mask.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            current_input++;
         }

      }


      DataSet ds = new DataSet( features, labels, features_mask, labels_mask );
      return ds;
   }


   public void Load( String file )
   {
      File f = new File( file );
      net = SerializationUtils.readObject( f );
   }


   private int getNextRelationID( Document doc )
   {
      LinkedHashSet< SemanticItem > sis = doc.getAllSemanticItems();
      int r_id = 0;
      for( SemanticItem si : sis )
      {
         if( si.getId().charAt( 0 ) == 'R' )
         {
            r_id = Integer.parseInt( si.getId().substring( 1 ) );
         }
      }
      r_id++;
      return r_id;
   }


   public void Run( String XML_Input_Dir )//, String XML_Output_Dir )
   {
      File in_dir = new File( XML_Input_Dir );

      for( File xml : in_dir.listFiles() )
      {
//         System.out.println( xml.getName() );

         Document doc = XMLToDoc( xml );

         List< SurfaceElement > words = doc.getAllSurfaceElements();
         LinkedHashSet< SemanticItem > terms = getTerms( doc );

         int relation_id = getNextRelationID( doc );

         for( SemanticItem c_si_antecedent : terms )
         {
            SurfaceElement c_antecedent_word = doc.getSurfaceElementsInSpan( c_si_antecedent.getSpan() ).get( 0 );

            for( SemanticItem c_si_anaphor : terms )
            {
               if( c_si_antecedent.equals( c_si_anaphor ) )
                  continue;

               SurfaceElement c_anaphor_word = doc.getSurfaceElementsInSpan( c_si_anaphor.getSpan() ).get( 0 );

               INDArray features = Nd4j.zeros( words.size() + 1, input_size );

               features.putScalar( new int[] {
                     words.size(),
                     do_output_index }, 1.0 );

               for( int word_num = 0; word_num < words.size(); word_num++ )
               {
                  SurfaceElement word = words.get( word_num );
                  boolean candidate_anaphor = c_anaphor_word.getSpan().equals( word.getSpan() );
                  boolean candidate_antecedent = c_antecedent_word.getSpan().equals( word.getSpan() );
                  boolean match_lemma_anaphor = c_anaphor_word.getLemma().equals( word.getLemma() );
                  boolean match_lemma_antecdent = c_antecedent_word.getLemma().equals( word.getLemma() );

                  String word_pos = cleanupPOS( word.getPos() );

                  int pos_index;
                  try
                  {
                     pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();
                  } catch( Exception e )
                  {
                     System.out.println( word.getPos() );
                     System.out.println( "ADD THIS" );
                     throw e;
                  }

                  if( candidate_anaphor )
                     features.put( new INDArrayIndex[] {
                           NDArrayIndex.point( word_num ),
                           NDArrayIndex.point( is_candidate_anaphor_index ) }, 1.0 );

                  if( candidate_antecedent )
                     features.put( new INDArrayIndex[] {
                           NDArrayIndex.point( word_num ),
                           NDArrayIndex.point( is_candidate_antecedent_index ) }, 1.0 );

                  if( match_lemma_anaphor )
                     features.put( new INDArrayIndex[] {
                           NDArrayIndex.point( word_num ),
                           NDArrayIndex.point( anaphor_match_index ) }, 1.0 );

                  if( match_lemma_antecdent )
                     features.put( new INDArrayIndex[] {
                           NDArrayIndex.point( word_num ),
                           NDArrayIndex.point( antecedent_match_index ) }, 1.0 );

                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( word_num ),
                        NDArrayIndex.point( pos_index ) }, 1.0 );
               }

//               System.out.println( input_size + " : " + words.size() + 1 );
               INDArray result = net.output( features );
//               for( int d : result.shape() )
//                  System.out.print( d + " " );
//               System.out.println();

               if( result.getDouble( words.size(), output_no_index, 0 ) <= result.getDouble( words.size(), output_yes_index, 0 ) )
               {
                  System.out.println( c_antecedent_word.getText() + " : " + c_antecedent_word.getSpan().toString() );
                  System.out.println( c_anaphor_word.getText() + " : " + c_anaphor_word.getSpan().toString() );
                  System.out.println();
               }
            }
         }
         System.out.println();
      }
   }


   public void TrainFromXMLDir( String XML_Dir, int total_epochs )
   {
      DataSet ds = XMLDirToDatSet( XML_Dir );
      net.setListeners( new ScoreIterationListener( 1 ) );

      for( int epoch = 0; epoch < total_epochs; epoch++ )
      {
         System.out.println( "Epoch: " + epoch );
         net.fit( ds );
      }
   }


   public void Save( String file )
   {
      File out = new File( file );
      SerializationUtils.saveObject( net, out );
   }

   private static String cleanupPOS( String pos )
   {
      pos = pos.replaceAll( "[-$]", "" );
      if( pos.equals( "." ) )
         pos = "PERIOD";
      if( pos.equals( "," ) )
         pos = "COMMA";
      if( pos.equals( ":" ) )
         pos = "COLON";
      if( pos.equals( "``" ) )
         pos = "DACCENT";
      if( pos.equals( "''" ) )
         pos = "DSQUOTE";
      return pos;
   }

   private static LinkedHashSet< SemanticItem > getAnaphoraRelations( Document doc )
   {
      LinkedHashSet< SemanticItem > s = (LinkedHashSet< SemanticItem >) doc.getAllSemanticItems().clone();
      s.removeIf( p -> !(p.getType().equals( "Anaphora" )) );
      return s;
   }

   private static LinkedHashSet< SemanticItem > getTerms( Document doc )
   {
      LinkedHashSet< SemanticItem > s = (LinkedHashSet< SemanticItem >) doc.getAllSemanticItems().clone();
      s.removeIf( p -> !(p.getId().charAt( 0 ) == 'T') );          // Doesn't seem to be a real way to get the terms
      return s;
   }
}
