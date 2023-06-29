// Helper function for more concise bincode definitions.
// They need to be manually kept up to date with the type.
//
// You might ask why this is needed. It turns out that deriving implementations is
// incompatible with specifying constraints on generic struct parameters, eg.
//   #[derive(bincode::Decode)]
//   struct SparseBitVec<Ones: BitBlock> { ... },
// will error even though constaining the Ones type is necessary to type some fields of that struct.
// There were also compilation errors that prevented the use of derive at the same time
// as specifing default values for generic struct parameters, eg.
//   struct Foo<T: BitBlock = u8> { ... }

// Fortuntely, the decode macros will throw an error if the fields or names change, which gives us an opportunity
// to also update the corresponding encode macro (which by itself could easily get out of sync).
// It would be nice to add construction and (de)serialization to the bitvec trait and test it automatically.

// fn encode<E: bincode::enc::Encoder>(
//     &self,
//     encoder: &mut E,
// ) -> core::result::Result<(), bincode::error::EncodeError> {
//     bincode::Encode::encode(&self.raw, encoder)?;
//     bincode::Encode::encode(&self.sr_pow2, encoder)?;
//     bincode::Encode::encode(&self.ss_pow2, encoder)?;
//     bincode::Encode::encode(&self.r, encoder)?;
//     bincode::Encode::encode(&self.s0, encoder)?;
//     bincode::Encode::encode(&self.s1, encoder)?;
//     bincode::Encode::encode(&self.num_ones, encoder)?;
//     Ok(())
// }
macro_rules! bincode_encode_impl {
    ($($t:ident),* $(,)?) => (
        fn encode<E: bincode::enc::Encoder>(
            &self,
            encoder: &mut E,
        ) -> core::result::Result<(), bincode::error::EncodeError> {
            $(bincode::Encode::encode(&self.$t, encoder)?;)*
            Ok(())
        }
    )
}

// Note: The macro assumes that the relevant generic lifetime is called 'de:
//   impl<'de> bincode::BorrowDecode<'de> for T { ... }
//
// This implements shorthand for:
// fn decode<D: bincode::de::Decoder>(
//     decoder: &mut D,
// ) -> core::result::Result<Self, bincode::error::DecodeError> {
//     Ok(Self {
//         raw: bincode::Decode::decode(decoder)?,
//         sr_pow2: bincode::Decode::decode(decoder)?,
//         ss_pow2: bincode::Decode::decode(decoder)?,
//         r: bincode::Decode::decode(decoder)?,
//         s0: bincode::Decode::decode(decoder)?,
//         s1: bincode::Decode::decode(decoder)?,
//         num_ones: bincode::Decode::decode(decoder)?,
//     })
// }
macro_rules! bincode_decode_impl {
    ($($t:ident),* $(,)?) => (
        fn decode<D: bincode::de::Decoder>(
            decoder: &mut D,
        ) -> core::result::Result<Self, bincode::error::DecodeError> {
            Ok(Self {
                $($t: bincode::Decode::decode(decoder)?,)*
            })
        }
    )
}

// fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
//     decoder: &mut D,
// ) -> core::result::Result<Self, bincode::error::DecodeError> {
//     Ok(Self {
//         raw: bincode::BorrowDecode::borrow_decode(decoder)?,
//         sr_pow2: bincode::BorrowDecode::borrow_decode(decoder)?,
//         ss_pow2: bincode::BorrowDecode::borrow_decode(decoder)?,
//         r: bincode::BorrowDecode::borrow_decode(decoder)?,
//         s0: bincode::BorrowDecode::borrow_decode(decoder)?,
//         s1: bincode::BorrowDecode::borrow_decode(decoder)?,
//         num_ones: bincode::BorrowDecode::borrow_decode(decoder)?,
//     })
// }
macro_rules! bincode_borrow_decode_impl {
    ($($t:ident),* $(,)?) => (
        fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
            decoder: &mut D,
        ) -> core::result::Result<Self, bincode::error::DecodeError> {
            Ok(Self {
                $($t: bincode::BorrowDecode::borrow_decode(decoder)?,)*
            })
        }
    )
}

pub(crate) use bincode_borrow_decode_impl;
pub(crate) use bincode_decode_impl;
pub(crate) use bincode_encode_impl;
