#include <stdio.h>
#include <math.h>
#define fracntion 18
#define word 32
#define BIT(a,bit) ((((signed long long)(a))&((signed long long)1 << (bit) )) ? 1: 0)
#define BITS(a, low, high) (((a)& (((signed long long)1 << high+1)-((signed long long)1<<low)))>>low)
#define CLIP_TC32(a, word_len) (BIT((a), 63) ? \
                        (BITS((a), (word_len), 62) != ((signed long long)1<<(63-(word_len))) - 1) ? \
                            (signed long long int)((signed long long int)(0x8000000000000000) >> (64-(word_len+1))) : (signed long long int)((signed long long int)(0x8000000000000000| ((signed long long int)(BITS(a, 0, word_len-1)) << (64-word_len-1))) >> (64-word_len-1)) \
                      : (BITS((a), (word_len), 62) != 0) ? \
                            ((long long int)1 << ((word_len))) -1 : (long long int)BITS(a, 0, word_len-1))
#define CLIP_ADD_TC32(a, b, word_len) (CLIP_TC32((a) + (b), (word_len)))
#define qadd(a,b) CLIP_ADD_TC32(a,b,word)
#define CLIP_MULT_TC32(a, b, word_len, fraction_len) (CLIP_TC32(((a) * (b))>>(fraction_len), (word_len)))
#define qmul(a,b) CLIP_MULT_TC32(a,b,word,fracntion)
#define FP2FXP_TC32(a, word_len, fraction_len) CLIP_TC32((signed long long int)(floorf((a) * powf(2.0F, (fraction_len)))), (word_len))
#define fp2fx(a) FP2FXP_TC32(a,word,fracntion)
#define FXP2FP_TC32(a, fraction_len) (BIT((a), 62) ? -(((float)(~(a)+1)) / powf(2.0F, (fraction_len))):(((float)(a)) / powf(2.0F, (fraction_len))))
#define fx2fp(a) FXP2FP_TC32(a,fracntion)


#define _QMATH_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef INLINE
#ifdef _MSC_VER
#define INLINE __inline
#else
#define INLINE inline
#endif /* _MSC_VER */
#endif /* INLINE */

	/*
	 * Default fractional bits. This precision is used in the routines
	 * and macros without a leading underscore.
	 * For example, if you are mostly working with values that come from
	 * a 10-bit A/D converter, you may want to choose 21. This leaves 11
	 * bits for the whole part, which will help avoid overflow in addition.
	 * On ARM, bit shifts require a single cycle, so all fracbits
	 * require the same amount of time to compute and there is no advantage
	 * to selecting fracbits that are a multiple of 8.
	 */
#define	FIXED_FRACBITS fracntion

#define FIXED_RESOLUTION (1 << FIXED_FRACBITS)
#define FIXED_INT_MASK (0xffffffffL << FIXED_FRACBITS)
#define FIXED_FRAC_MASK (~FIXED_INT_MASK)

	 // square roots
#define FIXED_SQRT_ERR 1 << (FIXED_FRACBITS - 4)

// fixedp2a
#define FIXED_DECIMALDIGITS word-fracntion

	typedef long long fixedp;

	// conversions for arbitrary fracbits
#define _short2q(x, fb)			((fixedp)((x) << (fb)))
#define _int2q(x, fb)			((fixedp)((x) << (fb)))
#define _long2q(x, fb)			((fixedp)((x) << (fb)))
#define _float2q(x, fb)			((fixedp)((x) * (1 << (fb))))
#define _double2q(x, fb)		((fixedp)((x) * (1 << (fb))))

// conversions for default fracbits
#define short2q(x)			_short2q(x, FIXED_FRACBITS)
#define int2q(x)			_int2q(x, FIXED_FRACBITS)
#define long2q(x)			_long2q(x, FIXED_FRACBITS)
#define float2q(x)			_float2q(x, FIXED_FRACBITS)
#define double2q(x)			_double2q(x, FIXED_FRACBITS)

// conversions for arbitrary fracbits	
#define _q2short(x, fb)		((short)((x) >> (fb)))
#define _q2int(x, fb)		((int)((x) >> (fb)))
#define _q2long(x, fb)		((long)((x) >> (fb)))
#define _q2float(x, fb)		((float)(x) / (1 << (fb)))
#define _q2double(x, fb)	((double)(x) / (1 << (fb)))

// conversions for default fracbits
#define q2short(x)			_q2short(x, FIXED_FRACBITS)
#define q2int(x)			_q2int(x, FIXED_FRACBITS)
#define q2long(x)			_q2long(x, FIXED_FRACBITS)
#define q2float(x)			_q2float(x, FIXED_FRACBITS)
#define q2double(x)			_q2double(x, FIXED_FRACBITS)

// evaluates to the whole (integer) part of x
#define qipart(x)			q2long(x)

// evaluates to the fractional part of x
#define qfpart(x)			((x) & FIXED_FRAC_MASK)

 // Both operands in addition and subtraction must have the same fracbits.
 // If you need to add or subtract fixed point numbers with different
 // fracbits, then use q2q to convert each operand beforehand.

/**
 * q2q - convert one fixed point type to another
 * x - the fixed point number to convert
 * xFb - source fracbits
 * yFb - destination fracbits
 */
	static INLINE fixedp q2q(fixedp x, int xFb, int yFb)
	{
		if (xFb == yFb) {
			return x;
		}
		else if (xFb < yFb) {
			return x << (yFb - xFb);
		}
		else {
			return x >> (xFb - yFb);
		}
	}

	/**
	 * Multiply two fixed point numbers with arbitrary fracbits
	 * x - left operand
	 * y - right operand
	 * xFb - number of fracbits for X
	 * yFb - number of fracbits for Y
	 * resFb - number of fracbits for the result
	 *
	 */
#define _qmul(x, y, xFb, yFb, resFb) ((fixedp)(((x) * (y)) >> ((xFb) + (yFb) - (resFb))))
#define qqmul(x,y) _qmul(x,y,fracntion,fracntion,fracntion)
	 /**
	  * Fixed point multiply for default fracbits.
	  *
	  *
	  *
	  /**
	   * divide
	   * shift into 64 bits and divide, then truncate
	   */
#define _qdiv(x, y, xFb, yFb, resFb) ((fixedp)(((x) << ((xFb) + (yFb) - (resFb))) / y))

	   /**
		* Fixed point divide for default fracbbits.
		*/
#define qdiv(x, y) _qdiv(x, y, FIXED_FRACBITS, FIXED_FRACBITS, FIXED_FRACBITS)

		/**
		 * Invert a number (x^-1) for arbitrary fracbits
		 */
#define _qinv(x, xFb, resFb) ((fixedp)(((1) << (xFb + resFb)) / x))

		 /**
		  * Invert a number with default fracbits.
		  */
#define qinv(x) _qinv(x, FIXED_FRACBITS, FIXED_FRACBITS);

		  /**
		   * Modulus. Modulus is only defined for two numbers of the same fracbits
		   */
#define qmod(x, y) ((x) % (y))

		   /**
			* Absolute value. Works for any fracbits.
			*/
#define qabs(x) (((x) < 0) ? (-x) : (x))

			/**
			 * Floor for arbitrary fracbits
			 */
#define _qfloor(x, fb) ((x) & (0xffffffff << (fb)))

			 /**
			  * Floor for default fracbits
			  */
#define qfloor(x) _qfloor(x, FIXED_FRACBITS)

			  /**
			   * ceil for arbitrary fracbits.
			   */
	static INLINE fixedp _qceil(fixedp x, int fb)
	{
		// masks off fraction bits and adds one if there were some fractional bits
		fixedp f = _qfloor(x, fb);
		if (f != x) return qadd(f, _int2q(1, fb));
		return x;
	}

	/**
	 * ceil for default fracbits
	 */
#define qceil(x) _qceil(x, FIXED_FRACBITS)

	 /**
	  * square root for default fracbits
	  */
	fixedp qsqrt(fixedp p_Square);

	/**
	 * exp (e to the x) for default fracbits
	 */
	fixedp qexp(fixedp p_Base);

	/**
	 * pow for default fracbits
	 */
	fixedp qpow(fixedp p_Base, fixedp p_Power);





	/**
	 * square root
	 */
	fixedp qsqrt(fixedp p_Square)
	{
		fixedp   res;
		fixedp   delta;
		fixedp   maxError;

		if (p_Square <= 0)
			return 0;

		/* start at half */
		res = (p_Square >> 1);

		/* determine allowable error */
		maxError = qmul(p_Square, FIXED_SQRT_ERR);

		do
		{
			delta = (qmul(res, res) - p_Square);
			res -= qdiv(delta, (res << 1));
		} while (delta > maxError || delta < -maxError);

		return res;
	}


	/**
	 * exp (e to the x)
	 */
	fixedp qexp(fixedp p_Base)
	{
		fixedp w;
		fixedp y;
		fixedp num;

		for (w = int2q(1), y = int2q(1), num = int2q(1); y != y + w; num += int2q(1))
		{
			w = qmul(w, qdiv(p_Base, num));
			y += w;
		}

		return y;
	}

	void binary(long long a) {
		for (int i = 63; i >= 0; i--) {
			printf("%d ", BIT(a, i));
		}
		printf("\n");
	}



