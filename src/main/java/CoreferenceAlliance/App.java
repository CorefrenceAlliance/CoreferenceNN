package CoreferenceAlliance;


import java.io.File;

public class App
{
   public static void main( String[] args )
   {
      LSTM lstm = new LSTM();

//      lstm.CreateNewNet();
//      lstm.TrainFromXMLDir( "DATA\\BIONLP\\Michael_XML", 1 );
//      lstm.Save( "D:\\large_nlp_mxe_1" );
//      lstm.TrainFromXMLDir( "DATA\\BIONLP\\Michael_XML", 1 );
//      lstm.Save( "D:\\large_nlp_mxe_2" );
//      lstm.TrainFromXMLDir( "DATA\\BIONLP\\Michael_XML", 1 );
//      lstm.Save( "D:\\large_nlp_mxe_3" );

      lstm.Load( "D:\\large_nlp_mxe_2" );
      lstm.Run( "DATA\\BIONLP\\test" );
   }
}